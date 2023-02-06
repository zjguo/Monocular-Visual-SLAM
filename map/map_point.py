from nptyping import NDArray, Shape, Float
import copy

class MapPoint():

    
    def __init__(self,
                id : int,
                point3d : NDArray[Shape["3,1"], Float],
                color : NDArray[Shape["3,1"], Float],
                drawn : bool = False):

        # for tracking
        self.id = id
        self.point3d = copy.deepcopy(point3d)
        self.frame_ids = set()

        # for visualization
        self.color = color

    def insert_frame_id(self, frame_id):
        self.frame_ids.add(frame_id)