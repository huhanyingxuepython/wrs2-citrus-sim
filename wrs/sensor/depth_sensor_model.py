"""GPU depth encoding, distortion lookup and point-cloud conversion."""

import numpy as np
import wgpu


_SHADER = """
struct Params {
    world_from_camera: mat4x4<f32>,
    sizes: vec4<u32>, // output width/height, source width/height
    range: vec4<f32>, // min, max, depth scale, fx * baseline
    noise: vec4<f32>,
    dropout: vec4<f32>,
    control: vec4<u32>,
};
struct Pixel {
    camera: vec4<f32>,
    world: vec4<f32>,
    ground_truth: f32,
    raw: u32,
    color: u32,
    confidence: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var z_image: texture_2d<f32>;
@group(0) @binding(2) var rgb_image: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> lut: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> pixels: array<Pixel>;
@group(0) @binding(5) var<storage, read_write> right_inverse_z: array<atomic<u32>>;

fn hash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}
fn random(index: u32, salt: u32) -> f32 {
    return (f32(hash(index ^ params.control.x ^ salt) >> 8u) + 0.5) / 16777216.0;
}
fn z_at(p: vec2<i32>) -> f32 {
    let clamped = clamp(p, vec2<i32>(0), vec2<i32>(params.sizes.zw)-1);
    return textureLoad(z_image, clamped, 0).r;
}
fn luminance(p: vec2<i32>) -> f32 {
    let clamped = clamp(p, vec2<i32>(0), vec2<i32>(params.sizes.zw)-1);
    return dot(textureLoad(rgb_image, clamped, 0).rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}
fn confidence_at(p: vec2<i32>) -> f32 {
    let a = luminance(p+vec2<i32>(-1,-1));
    let b = luminance(p+vec2<i32>(0,-1));
    let c = luminance(p+vec2<i32>(1,-1));
    let d = luminance(p+vec2<i32>(-1,0));
    let f = luminance(p+vec2<i32>(1,0));
    let g = luminance(p+vec2<i32>(-1,1));
    let h = luminance(p+vec2<i32>(0,1));
    let i = luminance(p+vec2<i32>(1,1));
    return clamp(length(vec2<f32>(c+2.0*f+i-a-2.0*d-g, g+2.0*h+i-a-2.0*b-c)), 0.0, 1.0);
}

// Reproject the visible left surface into a synthetic right image. Positive
// float bit patterns are ordered, so atomicMax(1/Z) retains the nearest point.
@compute @workgroup_size(8, 8) fn reproject(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sizes.z || id.y >= params.sizes.w) { return; }
    let z = z_at(vec2<i32>(id.xy));
    if (z <= 0.0) { return; }
    let right_u = i32(floor(f32(id.x)-params.range.w/z+0.5));
    if (right_u >= 0 && right_u < i32(params.sizes.z)) {
        atomicMax(&right_inverse_z[id.y*params.sizes.z+u32(right_u)], bitcast<u32>(1.0/z));
    }
}

fn sensor_z(p: vec2<i32>, gt: f32, index: u32) -> vec2<f32> {
    if (gt <= 0.0 || params.control.y == 0u) { return vec2<f32>(gt, 1.0); }
    let fb = params.range.w;
    if (params.control.z != 0u) {
        let right_u = i32(floor(f32(p.x)-fb/gt+0.5));
        if (right_u < 0 || right_u >= i32(params.sizes.z)) { return vec2<f32>(0.0); }
        let inverse = bitcast<f32>(atomicLoad(&right_inverse_z[u32(p.y)*params.sizes.z+u32(right_u)]));
        if (inverse <= 0.0 || gt > 1.0/inverse+params.dropout.w) { return vec2<f32>(0.0); }
    }
    var disparity = fb/gt;
    if (params.noise.x > 0.0) { disparity = round(disparity/params.noise.x)*params.noise.x; }
    if (params.noise.y > 0.0) {
        let gaussian = sqrt(-2.0*log(random(index, 0x43a9b5c1u))) * cos(6.28318530718*random(index, 0x91bc2d77u));
        disparity += params.noise.y*gaussian;
    }
    if (disparity <= 0.0) { return vec2<f32>(0.0); }
    var confidence = 1.0;
    if (params.noise.z > 0.0 || params.noise.w > 0.0) {
        confidence = confidence_at(p);
        if (confidence < params.noise.z || random(index, 0x163a2f55u) < params.noise.w*(1.0-confidence)) {
            return vec2<f32>(0.0, confidence);
        }
    }
    if (params.dropout.x > 0.0) {
        let jump = max(max(abs(gt-z_at(p+vec2<i32>(-1,0))), abs(gt-z_at(p+vec2<i32>(1,0)))),
                       max(abs(gt-z_at(p+vec2<i32>(0,-1))), abs(gt-z_at(p+vec2<i32>(0,1)))));
        if (jump > params.dropout.z && random(index, 0x772938a1u) < params.dropout.x) {
            return vec2<f32>(0.0, confidence);
        }
    }
    if (random(index, 0xa5bc1911u) < params.dropout.y) { return vec2<f32>(0.0, confidence); }
    return vec2<f32>(fb/disparity, confidence);
}

@compute @workgroup_size(8, 8) fn encode_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.sizes.x || id.y >= params.sizes.y) { return; }
    let index = id.y * params.sizes.x + id.x;
    let ray = lut[index];
    let source = vec2<i32>(ray.zw);
    let gt = textureLoad(z_image, source, 0).r;
    let rgb = textureLoad(rgb_image, source, 0);
    // Evaluate the sensor in the rectified source domain, then sample via the
    // distortion LUT. Pixels sharing a source sample share its noise as well.
    let sensor = sensor_z(source, gt, u32(source.y)*params.sizes.z+u32(source.x));
    let measured = sensor.x;
    var raw = 0u;
    // Allow only floating-point roundoff at inclusive measurement boundaries.
    let tolerance = 4.76837158203125e-7 * params.range.y;
    if (gt > 0.0 && measured > 0.0 &&
        gt >= params.range.x-tolerance && gt <= params.range.y+tolerance &&
        measured >= params.range.x-tolerance && measured <= params.range.y+tolerance) {
        let units = round(measured / params.range.z);
        let min_units = params.control.w & 65535u;
        let max_units = params.control.w >> 16u;
        if (units >= f32(min_units) && units <= f32(max_units)) {
            raw = u32(units);
        }
    }
    pixels[index].ground_truth = gt;
    pixels[index].raw = raw;
    pixels[index].color = pack4x8unorm(rgb);
    pixels[index].confidence = select(0.0, sensor.y, gt > 0.0);
    let z = f32(raw) * params.range.z;
    pixels[index].camera = vec4<f32>(ray.xy*z, z, 0.0);
    pixels[index].world = vec4<f32>(0.0);
    if (raw > 0u) {
        pixels[index].world = params.world_from_camera * vec4<f32>(ray.xy*z, z, 1.0);
    }
}
"""


class _GpuDepthProcessor:
    """One cached LUT upload, one packed readback for all frame outputs."""

    def __init__(self, device, model, depth_texture, rgb_texture):
        self.device, self.model = device, model
        self._size = model.width * model.height * 48
        source = model.render_model
        if max(self._size, source.width*source.height*4) > device.limits['max-storage-buffer-binding-size']:
            raise ValueError('camera output exceeds the GPU storage buffer limit')
        self._uniform = device.create_buffer(size=144,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._lut = device.create_buffer_with_data(
            data=np.ascontiguousarray(model.remap_lut, dtype=np.float32),
            usage=wgpu.BufferUsage.STORAGE)
        self._output = device.create_buffer(size=self._size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        self._readback = device.create_buffer(size=self._size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        self._right = device.create_buffer(size=source.width*source.height*4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        module = device.create_shader_module(code=_SHADER, label='depth_sensor_model')
        # r32float is unfilterable; the explicit layout avoids requiring float32 filtering.
        layout = device.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': 'uniform'}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
             'texture': {'sample_type': 'unfilterable-float'}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
             'texture': {'sample_type': 'float'}},
            {'binding': 3, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': 'read-only-storage'}},
            {'binding': 4, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': 'storage'}},
            {'binding': 5, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': 'storage'}}])
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[layout])
        self._pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={'module': module, 'entry_point': 'encode_depth'})
        self._reproject = device.create_compute_pipeline(layout=pipeline_layout,
            compute={'module': module, 'entry_point': 'reproject'})
        self._bind_group = device.create_bind_group(layout=layout, entries=[
            {'binding': 0, 'resource': {'buffer': self._uniform}},
            {'binding': 1, 'resource': depth_texture.create_view()},
            {'binding': 2, 'resource': rgb_texture.create_view()},
            {'binding': 3, 'resource': {'buffer': self._lut}},
            {'binding': 4, 'resource': {'buffer': self._output}},
            {'binding': 5, 'resource': {'buffer': self._right}}])

    def encode(self, encoder, camera_tf, camera):
        params = np.zeros(36, np.float32)
        params[:16] = camera_tf.T.ravel()
        source = self.model.render_model
        params.view(np.uint32)[16:20] = [self.model.width, self.model.height, source.width, source.height]
        params[20:24] = [camera.min_depth, camera.max_depth, camera.depth_scale,
                         self.model.fx * (camera.baseline_m or 0)]
        noise = camera.noise
        params[24:28] = [noise.disparity_step_px, noise.disparity_std_px,
                         noise.confidence_threshold, noise.texture_dropout_strength]
        params[28:32] = [noise.edge_dropout_strength, noise.dropout_rate,
                         noise.edge_threshold_m, noise.occlusion_tolerance_m]
        legacy_noise = camera.mode == 'd405_fast' and getattr(camera, 'noise_model', None) is None
        params.view(np.uint32)[32:36] = [camera._rng.integers(0, 2**32, dtype=np.uint32),
                                        legacy_noise, noise.stereo_occlusion,
                                        camera._raw_min | (camera._raw_max << 16)]
        self.device.queue.write_buffer(self._uniform, 0, params)
        if legacy_noise and noise.stereo_occlusion:
            encoder.clear_buffer(self._right)
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(self._reproject)
            compute.set_bind_group(0, self._bind_group)
            compute.dispatch_workgroups((source.width+7)//8, (source.height+7)//8)
            compute.end()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self._pipeline)
        compute.set_bind_group(0, self._bind_group)
        compute.dispatch_workgroups((self.model.width+7)//8, (self.model.height+7)//8)
        compute.end()
        encoder.copy_buffer_to_buffer(self._output, 0, self._readback, 0, self._size)

    def read(self):
        self._readback.map_sync(wgpu.MapMode.READ)
        try:
            # read_mapped(copy=True) already returns owned CPU memory; do not
            # copy the entire dense cloud twice before unmapping.
            data = np.frombuffer(self._readback.read_mapped(), np.float32)
        finally:
            self._readback.unmap()
        data = data.reshape(self.model.height, self.model.width, 12)
        bits = data.view(np.uint32)
        rgba = np.ascontiguousarray(bits[..., 10], dtype='<u4')
        rgb = rgba.view(np.uint8).reshape(self.model.height, self.model.width, 4)[..., :3].copy()
        return dict(depth_gt=data[..., 8], depth=data[..., 2],
                    depth_raw=bits[..., 9].astype(np.uint16), rgb=rgb,
                    points_cam_image=data[..., :3], points_world_image=data[..., 4:7],
                    confidence=data[..., 11])

    def close(self):
        for buffer in (self._uniform, self._lut, self._output, self._readback, self._right):
            buffer.destroy()
