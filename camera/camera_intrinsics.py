from typing import Tuple
import numpy as np

class CameraIntrinsics():
    
    def __init__(self, 
                 focal_length : float, 
                 principal_point : Tuple[float, float],
                 horz_bias_factor: float,
                 sheer_factor : int  = 1):

        self.focal_length = focal_length
        self.principal_point = principal_point
        self.horz_bias_factor = horz_bias_factor
        self.sheer_factor = sheer_factor
        self.K = np.array([[self.focal_length, self.focal_length*self.sheer_factor, self.principal_point[0]],
                           [0, self.focal_length*(self.horz_bias_factor), self.principal_point[1]],
                           [0, 0, 1]], dtype=float)