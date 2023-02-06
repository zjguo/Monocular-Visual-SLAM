import numpy as np
from nptyping import NDArray, Shape, Float
from .camera_intrinsics import CameraIntrinsics
from .camera_extrinsics import CameraExtrinsics

class Camera():
    
    def __init__(self, 
                 camera_intrinsics : CameraIntrinsics,
                 camera_extrinsincs : CameraExtrinsics):
                 
        self.focal_length = camera_intrinsics.focal_length
        self.principal_point = camera_intrinsics.principal_point
        self.horz_bias_factor = camera_intrinsics.horz_bias_factor
        self.translation = camera_extrinsincs.translation
        self.rotation = camera_extrinsincs.rotation
        self.K = camera_intrinsics.K
        
    def get_projection_matrix(self):
        
        translation_matrix = np.array([[1, 0, 0, -self.translation[0].item()],
                                       [0, 1, 0, -self.translation[1].item()],
                                       [0, 0, 1, -self.translation[2].item()]])
        return self.K @ self.rotation @ translation_matrix

    def direct_linear_transform(self, 
                                homo_point_coords : NDArray[Shape["4,1"], Float]) -> NDArray[Shape["3,1"], Float]:

        return self.get_projection_matrix() @ homo_point_coords
        