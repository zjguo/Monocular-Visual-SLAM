from video.frame_sequence import *
from feature_detection.detector import *
from photogrammetry.estimation import *
from photogrammetry.triangulation import *
from camera.camera import*
from camera.camera_intrinsics import*
from camera.camera_extrinsics import*
from nptyping import NDArray, Float, Shape
import numpy as np
from matplotlib import pyplot as plt
from visualization.pyplot_visualizer import *
from map.map import *
import copy


'''
TODO: 
- Map point culling by frame presence - DONE
- Map frame culling for redundant frames
- Search for new map points by reprojection 
- better average reprojection calculation - DONE
'''

class SLAM():
    
    def __init__(self, 
                 frame_seq : FrameSequence,
                 camera_intrinsic: CameraIntrinsics):
        self.frames = frame_seq.frames
        self.camera_intrinsic = camera_intrinsic
        self.map = Map(PyplotVisualizer(20))
        
    def run(self):
        prev_frame = next(self.frames)
        curr_frame = next(self.frames)
        current_camera_extrinsic = CameraExtrinsics(np.zeros((3,1)), np.diag(np.ones(3,))) # default world coords
        self.map.create_add_frame(Camera(self.camera_intrinsic, copy.deepcopy(current_camera_extrinsic)))
        while(prev_frame is not None and curr_frame is not None):
            width = curr_frame.shape[1]
            height = curr_frame.shape[0]
            kp1, kp2, des1, des2, matches = detect_and_match_pair(prev_frame, 
                                                      curr_frame,
                                                      n_features=8000,
                                                      fast_threshold=31,
                                                      score_type=0,
                                                      detector_type='gftt',
                                                      lowes_ratio=0.85)
            matched_points1 = get_matched_point_matrix(kp1, [i.queryIdx for i in matches])
            matched_points2 = get_matched_point_matrix(kp2, [i.trainIdx for i in matches])

            if len(self.map.map_frames) == 1:
                self.map.map_frames[-1].all_kpts = [x.pt for x in kp1]
                self.map.map_frames[-1].all_des = des1

            # estimate essential matrix and relative pose
            E, inliers = estimate_essential_matrix_opencv(matched_points1, matched_points2, self.camera_intrinsic)
            inlier_points1 = matched_points1[:, inliers]
            inlier_points2 = matched_points2[:, inliers]
            t, R = get_realizable_relpose_from_essential_matrix(inlier_points1, inlier_points2, self.camera_intrinsic, self.camera_intrinsic, E)

            # update camera extrinsics using relative pose
            prev_camera_extrinsic = copy.deepcopy(current_camera_extrinsic)
            camera1 = Camera(self.camera_intrinsic, copy.deepcopy(prev_camera_extrinsic))
            current_camera_extrinsic.translation += current_camera_extrinsic.rotation.T @ t # bring back translation to world coords
            current_camera_extrinsic.rotation = R @ current_camera_extrinsic.rotation
            camera2 = Camera(self.camera_intrinsic, current_camera_extrinsic)

            # get 3d world points using triangulation
            world_points, valid_world_point_indices, score  = triangulate_world_points(matched_points1, matched_points2, camera1, camera2, "LLS", 0.1)
            world_points_indicies_in_front_of_both_cameras = self.check_world_points_in_front_of_cameras(world_points, camera1, camera2)
            world_point_indicies_enough_parallax = self.check_world_points_enough_parallax(world_points, camera1, camera2, 0.1)
            valid_world_point_indices = valid_world_point_indices \
                                        & world_points_indicies_in_front_of_both_cameras \
                                        & world_point_indicies_enough_parallax \
                                        & inliers
            print("Detected {num_points} valid points.".format(num_points = np.sum(valid_world_point_indices)))
            if np.sum(valid_world_point_indices) == 0:
                prev_frame = curr_frame
                curr_frame = next(self.frames)
                continue
            
            # Update map
            frame_id = self.map.create_add_frame(copy.deepcopy(camera2))
            self.map.get_map_frame(frame_id).all_kpts = [x.pt for x in kp2]
            self.map.get_map_frame(frame_id).all_des = des2
            print("---------Frame {id}------------".format(id=frame_id))
            for i,valid in enumerate(valid_world_point_indices):
                if valid:
                    prev_matched_kp_loc = kp1[matches[i].queryIdx].pt
                    curr_matched_kp_loc = kp2[matches[i].trainIdx].pt
                    point_color = prev_frame[int(prev_matched_kp_loc[1]), int(prev_matched_kp_loc[0]),::-1]/255
                    matched_map_point_id = self.map.map_frames[-2].get_map_point_id_from_location(prev_matched_kp_loc)
                    map_point_id = self.map.incorporate_map_point(frame_id, 
                                                                  curr_matched_kp_loc,
                                                                  world_points[:, i],
                                                                  point_color, 
                                                                  matched_map_point_id)
                    if len(self.map.map_frames) == 2:
                        self.map.incorporate_map_point(0, 
                                                    prev_matched_kp_loc,
                                                    world_points[:, i],
                                                    point_color, 
                                                    map_point_id,
                                                    )
            num_new_associations = self.map.find_update_new_map_point_associations_by_reprojection(frame_id, width ,height, radius=2, hamming_threshold=0.4)
            print("Number of new associations by reprojection: {num}".format(num=num_new_associations))
            
            num_culled_points = self.map.cull_recent_points(num_consecutive_frames=2)
            print("Number of culled points: {num}. Remaining map points {num_points}.".format(num=num_culled_points, num_points=len(self.map.map_points)))

            # bundle_adjustment
            reproj_error_no_BA = self.map.get_avg_reprojection_error_per_frame()
            chi_sq = 0
            if frame_id % 10 == 0:
                chi_sq = self.map.full_bundle_adjust(25)
            else:
                chi_sq = self.map.local_bundle_adjustment(num_prev_frames=2, iterations=10)
            reproj_error_BA = self.map.get_avg_reprojection_error_per_frame()
            print("Reprojection error no BA: {noBA:.5f}, Reprojection error BA: {BA:.5f}, Chi: {chi}".format(noBA = reproj_error_no_BA, BA = reproj_error_BA, chi=chi_sq))

            # Visualize
            self.map.show()
            show_matches(curr_frame, kp1, kp2, matches, valid_world_point_indices)

            # set-up for next iteration
            prev_frame = curr_frame
            curr_frame = next(self.frames)
            current_camera_extrinsic = CameraExtrinsics(copy.deepcopy(self.map.get_map_frame(frame_id).camera.translation), copy.deepcopy(self.map.get_map_frame(frame_id).camera.rotation))

    def check_world_points_in_front_of_cameras(self,
                                               points3d : NDArray[Shape["3,Any"], Float],
                                               camera1 : Camera,
                                               camera2 : Camera) -> list:
        
        world_points_wrt_camera1 = camera1.rotation.T @ (points3d - camera1.translation)
        world_points_wrt_camera2 = camera2.rotation.T @ (points3d - camera2.translation)
        world_points_indicies_in_front_of_both_cameras =  (world_points_wrt_camera1[2,:] > 0) & (world_points_wrt_camera2[2,:] > 0)
        return world_points_indicies_in_front_of_both_cameras
    
    def check_world_points_enough_parallax(self,
                                           points3d : NDArray[Shape["3,Any"], Float],
                                           camera1 : Camera,
                                           camera2 : Camera,
                                           min_parallax_angle = 1 # degrees
                                           ) -> list:
        
        camera1_to_point = points3d - camera1.translation
        camera2_to_point = points3d - camera2.translation
        cos_angles = np.sum(camera1_to_point * camera2_to_point, axis=0)/(np.linalg.norm(camera1_to_point, axis=0) * np.linalg.norm(camera2_to_point, axis=0))
        return (cos_angles < np.cos(np.deg2rad(min_parallax_angle))) & (cos_angles > 0)
