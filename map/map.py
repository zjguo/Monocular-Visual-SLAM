from nptyping import NDArray, Shape, Float
from .map_frame import *
from photogrammetry.bundle_adjustment import *
from visualization.pyplot_visualizer import *
from photogrammetry.triangulation import calculate_reprojection_errors
import numpy as np
from scipy.spatial import cKDTree, distance
import cv2 as cv
import math

class Map():
    
    def __init__(self, pyplot_visualizer : PyplotVisualizer):
        
        self.frame_id_counter = -1 
        self.point_id_counter = -1
        self.map_frames = []
        self.map_points = []
        self.visualizer = pyplot_visualizer

    def create_add_frame(self,
                         camera : Camera) -> int:
        new_map_frame = MapFrame(self.generate_frame_id(), camera)
        self.map_frames.append(new_map_frame)
        return new_map_frame.id

    def incorporate_map_point(self,
                            frame_id : int,
                            point2d : NDArray[Shape["2,1"], Int],
                            point3d : NDArray[Shape["3,1"], Float],
                            color : NDArray[Shape["3,1"], Float],
                            matched_point_id : int) -> int:
        
        map_point = None
        if matched_point_id is None:
            map_point = MapPoint(self.generate_point_id(), point3d, color)
            self.map_points.append(map_point)
        else:
            map_point = self.get_map_point(matched_point_id)
            map_point.insert_frame_id(frame_id)
            self.get_map_frame(frame_id).insert_map_point(map_point, point2d)

        map_point.insert_frame_id(frame_id)
        self.get_map_frame(frame_id).insert_map_point(map_point, point2d)

        return map_point.id

    def generate_frame_id(self):
        self.frame_id_counter += 1
        return self.frame_id_counter

    def generate_point_id(self):
        self.point_id_counter += 1
        return self.point_id_counter

    def get_map_frame(self, id):
        frames = [i for i in self.map_frames if i.id == id]
        if len(frames) != 1:
            raise Exception("Error obtaining frame!")
        
        return frames[0]

    def get_map_point(self, id):
        points = [i for i in self.map_points if i.id==id]
        if len(points) != 1:
            raise Exception("Error obtaining point!")
        
        return points[0]

    def full_bundle_adjust(self, 
                           iterations : int = 10,
                           motion_only : bool = False) -> float:
        bundle_adjust = BundleAdjustment()

        # add frames
        for frame in self.map_frames:
            if frame.id == 0:
                bundle_adjust.add_pose(frame.id, frame.camera, fixed=True)
            else:
                bundle_adjust.add_pose(frame.id, frame.camera, fixed=motion_only)

        # add points
        for point in self.map_points:
            bundle_adjust.add_point(point.id, point.point3d)
            associated_frames = [self.map_frames[x] for x in point.frame_ids]

            # add edges
            for associated_frame in associated_frames:
                measurement = associated_frame.get_location_from_map_point_id(point.id)
                bundle_adjust.add_edge(point.id, associated_frame.id, measurement)
        
        # optimize
        if (bundle_adjust.gauge_freedom()):
            print("Found gauge freedom!")
        bundle_adjust.optimize(max_iterations=iterations)

        # extract results
        for frame in self.map_frames:
            frame.camera.translation = np.expand_dims(bundle_adjust.get_pose(frame.id).translation(), axis=1)
            frame.camera.rotation = copy.deepcopy(bundle_adjust.get_pose(frame.id).rotation().R).T
        
        for point in self.map_points:
            point.point3d = copy.deepcopy(bundle_adjust.get_point(point.id))

        return bundle_adjust.active_chi2()
    
    def local_bundle_adjustment(self,
                                iterations : int = 10,
                                num_prev_frames : int = 3,
                                motion_only : bool = False) -> float:
        
        if len(self.map_frames) <= num_prev_frames:
            return self.full_bundle_adjust()
        
        bundle_adjust = BundleAdjustment()
        local_frame_ids = [x.id for x in self.map_frames[-1:-(num_prev_frames + 1):-1]]
        local_map_point_ids = set()
        # add frames
        for frame in self.map_frames:
            if frame.id not in local_frame_ids:
                bundle_adjust.add_pose(frame.id, frame.camera, fixed=True)
            else:
                bundle_adjust.add_pose(frame.id, frame.camera, fixed=False)
                local_map_point_ids |= set([x[0] for x in frame.map_points_ids_with_point2d])

        # add points
        for point in self.map_points:
            associated_frames = []
            if point.id in local_map_point_ids:
                bundle_adjust.add_point(point.id, point.point3d)
            else:
                bundle_adjust.add_point(point.id, point.point3d, fixed=motion_only)
            associated_frames = [self.map_frames[x] for x in point.frame_ids]

            # add edges
            for associated_frame in associated_frames:
                measurement = associated_frame.get_location_from_map_point_id(point.id)
                bundle_adjust.add_edge(point.id, associated_frame.id, measurement)
        
        # optimize
        if (bundle_adjust.gauge_freedom()):
            print("Found gauge freedom!")
        bundle_adjust.optimize(max_iterations=iterations)

        # extract results
        for frame in self.map_frames:
            frame.camera.translation = np.expand_dims(bundle_adjust.get_pose(frame.id).translation(), axis=1)
            frame.camera.rotation = copy.deepcopy(bundle_adjust.get_pose(frame.id).rotation().R).T
        
        for point in self.map_points:
            point.point3d = copy.deepcopy(bundle_adjust.get_point(point.id))

        return bundle_adjust.active_chi2()

    def show(self):

        self.visualizer.clear()
        # draw map frame cameras
        for map_frame in self.map_frames:
            self.visualizer.draw_camera(map_frame.camera.translation, map_frame.camera.rotation.T, scale=1)

        # draw map points
        point_mat = np.array([map_point.point3d for map_point in self.map_points]).T
        point_colors = [map_point.color for map_point in self.map_points]

        if point_mat.size != 0:
            self.visualizer.scatter(point_mat[0,:], point_mat[1,:], point_mat[2,:], s=1, alpha=0.75, colors=point_colors)
    
        plt.pause(0.01)

    def get_avg_reprojection_error_per_frame(self) -> Int:
        avg_error = 0
        for frame in self.map_frames:
            relevent_map_points = [x for x in self.map_points if frame.id in x.frame_ids]
            relevent_map_point_ids = [x.id for x in relevent_map_points]
            map_point_locations = [frame.get_location_from_map_point_id(x) for x in relevent_map_point_ids]
            avg_error += np.sum(calculate_reprojection_errors(np.array([x.point3d for x in relevent_map_points]).T, 
                                                              np.squeeze(np.array(map_point_locations)).T,
                                                              frame.camera)) / (len(self.map_frames) * len(relevent_map_point_ids))

        return avg_error           

    def remove_map_point(self, id):
        for i,p in enumerate(self.map_points):
            if p.id == id:
                for frame in [self.get_map_frame(x) for x in p.frame_ids]:
                    element = [x for x in frame.map_points_ids_with_point2d if x[0] == id]
                    frame.map_points_ids_with_point2d.remove(element[0])
                self.map_points.pop(i)
                return

    def cull_recent_points(self,
                           num_consecutive_frames = 3) -> int:
        if len(self.map_frames) < num_consecutive_frames:
            return 0
        
        prev_ids = [x.id for x in self.map_frames[-1:-num_consecutive_frames-1:-1]]

        num_points_culled = 0
        for p in [x for x in self.map_points if prev_ids[num_consecutive_frames-1] == min(x.frame_ids)]:
            check = [x in p.frame_ids for x in prev_ids[:num_consecutive_frames-1]]
            if not all(check):
                self.remove_map_point(p.id)
                num_points_culled += 1
        
        return num_points_culled
    
    def find_update_new_map_point_associations_by_reprojection(self,
                                                                frame_id : int,
                                                                img_width : int,
                                                                img_height: int,
                                                                radius : float = 2,
                                                                hamming_threshold : float = 0.5) -> int:
        
        frame = self.get_map_frame(frame_id)
        frame_locs_kd_tree = cKDTree(frame.all_kpts)
        num_new_associations = 0
        for p in self.map_points:
            if frame.id in p.frame_ids:
                continue

            homo_point3d = np.concatenate((p.point3d, [1]), axis=0)
            homo_point2d = frame.camera.direct_linear_transform(homo_point3d)
            point2d = homo_point2d[0:2]/homo_point2d[-1]

            # check if out of bounds
            within_width = (point2d[0] >= 0) & (point2d[0] <= img_width)
            within_height = (point2d[1] >= 0) & (point2d[1] <= img_height)
            if (not within_width) or (not within_height):
                continue

            near_features = [(frame.all_kpts[x], frame.all_des[x]) for x in frame_locs_kd_tree.query_ball_point(point2d, radius)]
            rep_descriptor = self.get_representative_map_point_descriptor(p.id)
            if len(near_features) == 0:
                continue
            else: 
                hamming_distances = [distance.hamming(x[1], rep_descriptor) for x in near_features]
                min_distance = min(hamming_distances)
                if min_distance > hamming_threshold:
                    continue
                min_idx = hamming_distances.index(min_distance)
                self.incorporate_map_point(frame.id, 
                            near_features[min_idx][0],
                            p.point3d,
                            p.color, 
                            p.id)
                num_new_associations += 1

        return num_new_associations
    
    def get_representative_map_point_descriptor(self,
                                                point_id):
        point = self.get_map_point(point_id)
        descriptors = []
        for frame in [self.get_map_frame(x) for x in point.frame_ids]:
            descriptors.append(frame.get_descriptor_from_map_point_id(point.id))

        sum_hamming_distances = [np.sum([distance.hamming(x, des) for x in descriptors]) for des in descriptors]
        min_idx = sum_hamming_distances.index(min(sum_hamming_distances))
        return descriptors[min_idx]
        



                



        


        