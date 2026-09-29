# Lab citrus reference fitting — V1, static

## Evidence reviewed

Source: `940474370c1c222d6a464947c9f1c92b.mp4`, 720 × 1280, 30 fps, 1,709 frames,
approximately 56.97 seconds. Reviewed frames at 0.00, 5.18, 10.35, 15.53, 20.70,
25.88, 31.05, 36.23, 41.41, 46.58, 51.76 and 56.93 seconds. Enlarged early,
middle and late frames were used to distinguish persistent foliage from robot
occlusion and fruit removal. This is visual/manual fitting, not calibrated 3D reconstruction.

| Time range | Useful observation |
| --- | --- |
| 0–5 s | Three distinct visible fruits; lower-right fruit at the gripper, upper-right fruit crossed by a hanging leaf, another fruit at middle-left. |
| 10–15.5 s | Lower-right fruit has been removed; leaves and the right-hand entry area become clearer. |
| 20.7–31 s | Upper-right fruit is approached and removed; branches/leaves move, so their instantaneous pose is not a static rest measurement. |
| 36–41.4 s | Only the middle-left fruit remains; upper/middle foliage and its dark interior can be checked without the former fruit. |
| 46.6–56.9 s | Final visible fruit is picked; lower hanging foliage and right-side twig boundary remain visible. |

The preset represents the **initial three visible fruits**, not a sum of fruit
detections across the clip. Hidden rear fruits cannot be counted reliably.

At t=0, manually estimated image centres (x, y pixels) are approximately
`orange_000=(384,838)`, `orange_001=(384,566)`, `orange_002=(155,696)`.
These observations are stored in the preset's `reference` metadata and are
not camera-calibrated 3D measurements.

## What was fitted

- A tall, irregular crown, heavier to the left of the assumed trunk; a higher
  left/central tip and lower right shoulder. The left edge is cropped in the video.
- Dense upper/middle foliage and a darker occupied interior, instead of the
  placeholder's exposed scaffold and broad empty bands. Lower outer shoots have
  fewer leaves and strongly downward pitch. All procedural leaves remain attached
  to terminal thin branches; none are sampled in a bounding box.
- Small right-side openings near the picking region. The preset does not carve
  a large empty tunnel through the canopy. Leaf gaps do not remove hard obstacles.
- Narrower pointed blades, modest fold/curl/droop and three green shades.
  WRS's existing matte rendering is retained; video gloss/lighting is not reproduced.
- Three fixed fruit attachments, ordered lower-right, upper-right, middle-left.
  Two explicitly attached leaves preserve the most obvious partial occlusions.
- Thin green/grey shoots and a restrained brown/green hidden scaffold. Invisible
  branches form a plausible connected graph rather than purported recovered anatomy.

Generic mechanisms live in generator/spec/demo code. Every coordinate, growth
profile, opening, authored leaf, colour and camera pose specific to this lab tree
lives in `configs/lab_tree_v1.json`. The old placeholder remains available as
`configs/generic_citrus_v1.json`.

## Nominal model measurements (not real-tree measurements)

At seed 20260916 and `scale_multiplier=1.0`:

| Quantity | Generated model |
| --- | --- |
| Root to top | 1.176 m |
| Tree AABB min / max | (-0.419, -0.341, -0.014) / (0.329, 0.289, 1.176) m |
| Tree AABB width / depth / height | 0.747 / 0.629 / 1.190 m |
| Leaf canopy AABB min / max | (-0.419, -0.341, 0.154) / (0.329, 0.289, 1.176) m |
| Leaf canopy height | 1.022 m |
| Branch segments | 274: 82 authored scaffold/support segments + 192 procedural twigs |
| Collidable branches | 17 capsules |
| Leaves | 1,528 in 3 mesh batches; count tuned for occupancy, not counted from video |
| Fruits | 3 individually accessible SceneObjects, each with sphere collision |

The negative tree AABB minimum Z is the conservative trunk capsule's end cap,
not a shifted root. TreeSpec root remains exactly (0,0,0).

| ID | Nominal tree-local centre (m) | Radius (m) | Video association |
| --- | --- | --- | --- |
| orange_000 | (0.095, -0.235, 0.535) | 0.039 | Lower-right, picked first |
| orange_001 | (0.070, -0.245, 0.770) | 0.042 | Upper-right, partly leaf-occluded, picked second |
| orange_002 | (-0.120, -0.180, 0.650) | 0.038 | Middle-left, picked last |

+X is screen-right in the approximate front view, -Y is the camera/accessible
side, +Z is up. Front/back offsets are assumptions. The fruit supports are
authored macro segments, so procedural seed/density changes cannot move targets.

Leaf allocation: upper/apex 453, middle-dense 396, interior 405, lower 240,
right-open 32, plus 2 authored occluding leaves. These are profile totals,
not an image segmentation or measured biological distribution.

## Scale and uncertainty

No verified robot/gripper dimension, physical orange diameter, camera calibration,
or measurement marker is available. The working scale assumes a mean orange
diameter near **0.080 m**, with individual nominal diameters 0.078/0.084/0.076 m.
The robot's appearance alone was not used to assert a known dimension.

`scale_multiplier` scales all positions, radii, leaf sizes/thickness and stems;
the fixed cameras and ground scale with it when starting the demo. If a known
fruit is measured, use `measured_diameter / that_fruit_nominal_diameter`.
Dimensions remain estimated until an actual metric reference is supplied.

Confidence is higher for visible fruit count/order, relative image positions,
leaf shape, upper/middle density and the hanging lower silhouette. Confidence
is lower for trunk position, complete left boundary, root-to-top height and
hidden branch directions. Canopy depth, rear foliage/fruit inventory and the
exact geometry of robot-access channels are particularly underconstrained:
the video does not provide a calibrated side view, and foliage moves during picking.

The metal support frame, robot/gripper, cables and laboratory background are
not represented by this TreeSpec. Main branch capsules are an inferred obstacle
model; they are not measured clearance guarantees for a real robot.

## Most useful measurements next

1. Diameter of one identified fruit (preferably orange_001), plus one vertical
   dimension such as root-to-tip or root-to-that-fruit centre. This anchors scale
   and tests the assumed root position.
2. Maximum canopy width and front-to-back thickness, and a side/front photo with
   a ruler or known baseline. This constrains the currently guessed Y geometry.
3. Each target centre relative to the trunk (X/Y/Z), with visible stem attachment.
   This improves approach planning more than reconstructing unseen fine twigs.
4. Trunk diameter and the positions/diameters of the few main branches bordering
   the actual gripper entry region; also gripper width/depth if corridor clearance
   is to be evaluated later.

Use the fixed views, layer selector, config watching and JSON reporting workflow
in README to iterate. No dynamics, spring joints or articulated simulation were added.
