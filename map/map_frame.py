from .map_point import *
from nptyping import NDArray, Shape, Int
from camera.camera import *
import cv2 as cv

class MapFrame():
    
    
    def __init__(self,
                id : int,
                camera : Camera,
                drawn : bool = False):

        # for tracking
        self.id = id
        self.camera = camera
        self.all_kpts = None
        self.all_des = None
        self.map_points_ids_with_point2d = set()

    def insert_map_point(self, map_point : MapPoint, point2d : NDArray[Shape["2,1"], Int]):
        self.map_points_ids_with_point2d.add((map_point.id, point2d))

    def get_map_point_id_from_location(self, location : NDArray[Shape["2,1"], Int]) -> int:
        id = [x[0] for x in self.map_points_ids_with_point2d if x[1] == location]
        if len(id) == 0:
            return None
        else:
            return id[0]
    
    def get_location_from_map_point_id(self, id : int) -> NDArray[Shape["2,1"], Int]:
        location = [x[1] for x in self.map_points_ids_with_point2d if x[0] == id]
        if len(location) == 0:
            return None
        else:
            return location[0]
        
    def get_descriptor_from_map_point_id(self, id):
        loc = self.get_location_from_map_point_id(id)
        idx = [i for i,x in enumerate(self.all_kpts) if x == loc]
        if len(idx) == 0:
            return None
        
        return self.all_des[idx[0]]
    
    def draw_map_point_locs(self, img):
        kpts = [cv.KeyPoint(x[1][0], x[1][1], size=1) for x in self.map_points_ids_with_point2d]
        output_image = cv.drawKeypoints(img, kpts, 0, (255, 0, 0),
                    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        output_image = cv.resize(output_image, (640,360))
        cv.imshow("map frame", output_image)
        cv.waitKey(25)

