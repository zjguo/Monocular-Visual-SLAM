from video.frame_sequence import *
from feature_detection.detector import *
from photogrammetry.estimation import *
from photogrammetry.triangulation import *
from camera.camera import*
from camera.camera_intrinsics import*
from camera.camera_extrinsics import*
import numpy as np
from visualization.pyplot_visualizer import *
from map.map import *
import copy

class SFM():
    
    def __init__(self, 
                 frame_seq : FrameSequence,
                 camera_intrinsic: CameraIntrinsics):
        self.frames = frame_seq.frames
        self.camera_intrinsic = camera_intrinsic
        self.map = Map(PyplotVisualizer(20))

    def run(self):
        prev_frame = next(self.frames)
        curr_frame = next(self.frames)
        current_camera_extrinsic = CameraExtrinsics(np.zeros((3,1)), np.diag(np.ones(3,)))
        self.map.create_add_frame(Camera(self.camera_intrinsic, copy.deepcopy(current_camera_extrinsic)))
        if(prev_frame is not None and curr_frame is not None):
            kp1, kp2, matches = detect_and_match_pair(prev_frame, 
                                                      curr_frame,
                                                      n_features=100000,
                                                      fast_threshold=0,
                                                      score_type=0,
                                                      detector_type='orb',
                                                      lowes_ratio=1)
            matched_points1 = get_matched_point_matrix(kp1, [i.queryIdx for i in matches])
            matched_points2 = get_matched_point_matrix(kp2, [i.trainIdx for i in matches])

            # estimate essential matrix and relative pose
            E, inliers = estimate_essential_matrix_opencv(matched_points1, matched_points2, self.camera_intrinsic)
            inlier_points1 = matched_points1[:, inliers]
            inlier_points2 = matched_points2[:, inliers]
            # t is wrt to previous (unrotated) camera coords
            t, R = get_realizable_relpose_from_essential_matrix(inlier_points1, inlier_points2, self.camera_intrinsic, self.camera_intrinsic, E)

            # update camera extrinsics using relative pose
            prev_camera_extrinsic = copy.deepcopy(current_camera_extrinsic)
            camera1 = Camera(self.camera_intrinsic, copy.deepcopy(prev_camera_extrinsic))
            current_camera_extrinsic.translation += current_camera_extrinsic.rotation.T @ t # bring back translation to world coords
            current_camera_extrinsic.rotation = R @ current_camera_extrinsic.rotation
            camera2 = Camera(self.camera_intrinsic, current_camera_extrinsic)

            # get 3d world points using triangulation
            world_points, valid_world_point_indices, score  = triangulate_world_points(matched_points1, matched_points2, camera1, camera2, "LLS", 10)
            world_points_wrt_camera1 = camera1.rotation.T @ (world_points - camera1.translation)
            world_points_wrt_camera2 = camera2.rotation.T @ (world_points - camera2.translation)
            world_points_indicies_in_front_of_both_cameras =  (world_points_wrt_camera1[2,:] > 0) & (world_points_wrt_camera2[2,:] > 0)
            valid_world_point_indices = valid_world_point_indices & world_points_indicies_in_front_of_both_cameras
            
            # Update map
            frame_id = self.map.create_add_frame(copy.deepcopy(camera2))
            for i,valid in enumerate(valid_world_point_indices):
                if valid:
                    prev_matched_kp_loc = kp1[matches[i].queryIdx].pt
                    curr_matched_kp_loc = kp2[matches[i].trainIdx].pt
                    point_color = prev_frame[int(prev_matched_kp_loc[1]), int(prev_matched_kp_loc[0]),::-1]/255
                    map_point_id = self.map.incorporate_map_point(frame_id, 
                                                curr_matched_kp_loc,
                                                world_points[:, i],
                                                point_color, None)
                    self.map.incorporate_map_point(0, 
                                                prev_matched_kp_loc,
                                                world_points[:, i],
                                                point_color, 
                                                map_point_id)

            # bundle_adjustment
            reproj_error_no_BA = self.map.get_avg_reprojection_error_per_frame()
            self.map.full_bundle_adjust()
            reproj_error_BA = self.map.get_avg_reprojection_error_per_frame()
            print("Reprojection error no BA: {noBA:.5f}, Reprojection error BA: {BA:.5f}".format(noBA = reproj_error_no_BA, BA = reproj_error_BA))

            # Visualize
            show_matches(curr_frame, kp1, kp2, matches, valid_world_point_indices)
            self.map.show()
            plt.pause(0.01)
            