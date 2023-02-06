import numpy as np
from camera.camera import *
import sys
import os
import copy
import g2o

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, camera : Camera, fixed=False):
        cam = g2o.SBACam(g2o.SE3Quat(copy.deepcopy(camera.rotation.T), np.squeeze(camera.translation)))
        cam.set_cam(camera.focal_length, camera.focal_length*camera.horz_bias_factor, camera.principal_point[0], camera.principal_point[1], 0)

        v_cam = g2o.VertexCam()
        v_cam.set_id(pose_id * 2)   # internal id
        v_cam.set_estimate(cam)
        v_cam.set_fixed(fixed)
        super().add_vertex(v_cam)

    def add_point(self, point_id, point_3d, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point_3d)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()