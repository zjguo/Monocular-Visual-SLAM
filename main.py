from applications.slam import *
from camera.camera_intrinsics import *


frames_seq = FrameSequence.from_mp4("test_data/SLAM/videos/test_kitti984.mp4")
camera_intrinsics = CameraIntrinsics(
    focal_length=984,
    principal_point=(618, 183),
    horz_bias_factor=1,
    sheer_factor=0)


slam = SLAM(frames_seq, camera_intrinsics)
slam.run()