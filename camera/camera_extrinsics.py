from nptyping import NDArray, Shape, Float
from typing import Tuple

class CameraExtrinsics():

    def __init__(self, translation: NDArray[Shape["3, 1"], Float], rotation : NDArray[Shape["3, 3"], Float]):
        self.translation = translation
        self.rotation = rotation