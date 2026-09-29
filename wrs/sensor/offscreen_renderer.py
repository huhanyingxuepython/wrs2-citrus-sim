"""WGPU triangle rasterization into metric Z and optional matte RGB attachments."""

import numpy as np
import wgpu
from time import perf_counter

from wrs.utils.gpu_device import get_device


_SHADER = """
struct Params {
    camera_from_model: mat4x4<f32>,
    intrinsics: vec4<f32>,
    projection: vec4<f32>,
    color: vec4<f32>,
    key_light: vec4<f32>, // optical direction + linear intensity
    fill_light: vec4<f32>,
    lighting: vec4<f32>, // ambient, key power, enabled, unused
};
@group(0) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) camera_point: vec3<f32>,
};
@vertex fn vs_main(@location(0) position: vec3<f32>) -> VertexOutput {
    let p = params.camera_from_model * vec4<f32>(position, 1.0);
    let k = params.intrinsics;
    let q = params.projection; // width, height, far/(far-near), far*near/(far-near)
    var out: VertexOutput;
    // WGPU pixel centers are half-integers; calibrated pixel centers are integers.
    out.clip = vec4<f32>(2.0 * (k.x*p.x + (k.z+0.5)*p.z) / q.x - p.z,
                        p.z - 2.0 * (k.y*p.y + (k.w+0.5)*p.z) / q.y,
                        q.z*p.z - q.w, p.z);
    out.camera_point = p.xyz;
    return out;
}
struct FragmentOutput {
    @location(0) depth_m: f32,
    @location(1) color: vec4<f32>,
};

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
                  c / 12.92, c <= vec3<f32>(0.04045));
}
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055),
                  c * 12.92, c <= vec3<f32>(0.0031308));
}

@fragment fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    // Rasterizers snap vertices to a subpixel grid. Recover the triangle plane
    // from interpolated camera positions, then evaluate Z on the exact optical
    // pixel ray to avoid amplified interpolation errors on oblique surfaces.
    let normal = cross(dpdx(in.camera_point), dpdy(in.camera_point));
    let k = params.intrinsics;
    let ray = vec3<f32>((in.clip.x - 0.5 - k.z) / k.x,
                        (in.clip.y - 0.5 - k.w) / k.y, 1.0);
    let denominator = dot(normal, ray);
    out.depth_m = select(in.camera_point.z, dot(normal, in.camera_point) / denominator,
                        abs(denominator) > 1e-20);
    out.color = params.color;
    if (params.lighting.z > 0.0) {
        // Face the visible side toward the optical centre, including reverse
        // leaf faces. Lighting never changes the geometric Z attachment.
        var n = normal / max(length(normal), 1e-20);
        n = select(n, -n, dot(n, in.camera_point) > 0.0);
        let key = pow(clamp(dot(n, params.key_light.xyz) * 0.5 + 0.5, 0.0, 1.0), params.lighting.y);
        let fill = max(dot(n, params.fill_light.xyz), 0.0);
        let intensity = params.lighting.x + key * params.key_light.w + fill * params.fill_light.w;
        out.color = vec4<f32>(linear_to_srgb(clamp(srgb_to_linear(params.color.rgb) * intensity,
                                                 vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
    }
    return out;
}
"""


class _OffscreenRenderer:
    """Own reusable attachments and GPU buffers on WRS's shared device.

    Visual geometry is treated as immutable, as in the viewer. Replacing a
    model's geometry is supported; editing vertices in place requires close()
    on the camera to invalidate its GPU cache. No browser/window is created.
    """

    def __init__(self, camera_model):
        self.device = get_device()
        self.model = camera_model.render_model
        if max(self.model.width, self.model.height) > self.device.limits['max-texture-dimension-2d']:
            raise ValueError('camera/distortion render target exceeds GPU texture limits')
        self._meshes = {}
        self._draws = {}
        module = self.device.create_shader_module(code=_SHADER, label='virtual_depth_camera')
        self.pipeline = self.device.create_render_pipeline(
            layout='auto',
            vertex={'module': module, 'entry_point': 'vs_main', 'buffers': [
                {'array_stride': 12, 'step_mode': 'vertex', 'attributes': [
                    {'shader_location': 0, 'offset': 0, 'format': 'float32x3'}]}]},
            primitive={'topology': 'triangle-list', 'cull_mode': 'none'},
            depth_stencil={'format': 'depth32float', 'depth_write_enabled': True,
                           'depth_compare': 'less-equal'},
            fragment={'module': module, 'entry_point': 'fs_main',
                      'targets': [{'format': 'r32float'}, {'format': 'rgba8unorm'}]})
        size = (self.model.width, self.model.height, 1)
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        self.depth = self.device.create_texture(size=size, format='r32float', usage=usage)
        self.rgb = self.device.create_texture(size=size, format='rgba8unorm', usage=usage)
        self.zbuffer = self.device.create_texture(
            size=size, format='depth32float', usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        from .depth_sensor_model import _GpuDepthProcessor
        self._processor = _GpuDepthProcessor(self.device, camera_model, self.depth, self.rgb)

    def render(self, objects, camera_tf, near, far, camera):
        started = perf_counter()
        camera_from_world = np.linalg.inv(camera_tf)
        encoder = self.device.create_command_encoder()
        light = camera.rgb_lighting
        light_uniforms = ([*light.key_direction, light.key_intensity,
                           *light.fill_direction, light.fill_intensity,
                           light.ambient, light.key_power, 1., 0.] if light else [0.] * 12)
        render_pass = encoder.begin_render_pass(
            color_attachments=[{'view': texture.create_view(), 'resolve_target': None,
                                'clear_value': clear,
                                'load_op': 'clear', 'store_op': 'store'}
                               for texture, clear in ((self.depth, (0, 0, 0, 0)),
                                                      (self.rgb, (*camera.rgb_background, 1)))],
            depth_stencil_attachment={'view': self.zbuffer.create_view(),
                                      'depth_clear_value': 1.0,
                                      'depth_load_op': 'clear', 'depth_store_op': 'store'})
        render_pass.set_pipeline(self.pipeline)
        active_meshes, active_draws = set(), set()
        k = self.model
        for owner in objects:
            camera_from_owner = camera_from_world @ owner.tf
            for index, model in enumerate(owner.visuals):
                geom = model.geom
                # Point clouds and collision debug meshes do not define sensor surfaces.
                if geom is None or geom.fs is None or len(geom.fs) == 0 or model.alpha <= 0:
                    continue
                active_meshes.add(geom)
                if geom not in self._meshes:
                    vertices = self.device.create_buffer_with_data(
                        data=np.ascontiguousarray(geom.vs, dtype=np.float32),
                        usage=wgpu.BufferUsage.VERTEX)
                    indices = self.device.create_buffer_with_data(
                        data=np.ascontiguousarray(geom.fs, dtype=np.uint32),
                        usage=wgpu.BufferUsage.INDEX)
                    self._meshes[geom] = (vertices, indices, geom.fs.size)
                key = (owner, index)
                active_draws.add(key)
                if key not in self._draws:
                    uniform = self.device.create_buffer(
                        size=160, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
                    bind_group = self.device.create_bind_group(
                        layout=self.pipeline.get_bind_group_layout(0),
                        entries=[{'binding': 0, 'resource': {'buffer': uniform}}])
                    self._draws[key] = (uniform, bind_group)
                uniform, bind_group = self._draws[key]
                data = np.concatenate(((camera_from_owner @ model.loc_tf).T.ravel(),
                                       [k.fx, k.fy, k.cx, k.cy],
                                       [k.width, k.height, far / (far-near), far*near / (far-near)],
                                       [*model.rgb, 1.0], light_uniforms)).astype(np.float32)
                self.device.queue.write_buffer(uniform, 0, data)
                vertices, indices, count = self._meshes[geom]
                render_pass.set_bind_group(0, bind_group)
                render_pass.set_vertex_buffer(0, vertices)
                render_pass.set_index_buffer(indices, 'uint32')
                render_pass.draw_indexed(count)
        render_pass.end()
        rendered = perf_counter()
        self._processor.encode(encoder, camera_tf, camera)
        processed = perf_counter()
        self.device.queue.submit([encoder.finish()])
        submitted = perf_counter()
        result = self._processor.read()
        self.timings_ms = {'render_encode': (rendered - started) * 1000,
                           'sensor_pointcloud_encode': (processed - rendered) * 1000,
                           'submit': (submitted - processed) * 1000,
                           'readback_wait': (perf_counter() - submitted) * 1000}
        for geom in self._meshes.keys() - active_meshes:
            vertices, indices, _ = self._meshes.pop(geom)
            vertices.destroy()
            indices.destroy()
        for key in self._draws.keys() - active_draws:
            self._draws.pop(key)[0].destroy()
        return result

    def close(self):
        for vertices, indices, _ in self._meshes.values():
            vertices.destroy()
            indices.destroy()
        for uniform, _ in self._draws.values():
            uniform.destroy()
        self._meshes.clear()
        self._draws.clear()
        self._processor.close()
        for resource in (self.depth, self.rgb, self.zbuffer):
            resource.destroy()
