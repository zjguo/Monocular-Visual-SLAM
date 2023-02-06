from nptyping import NDArray, Float, Shape
import numpy as np
from typing import Tuple
import math
from camera.camera import *
from camera.camera_intrinsics import *
from camera.camera_extrinsics import *
from sklearn.preprocessing import normalize
from .triangulation import *
import cv2 as cv

def estimate_essential_matrix_opencv(points1 : NDArray[Shape["3,Any"], Float],
                                    points2 : NDArray[Shape["3,Any"], Float],
                                    camera_intrinsics : CameraIntrinsics) -> Tuple[NDArray[Shape["3,3"], Float], list[int]]:
                              
    points1 = np.ascontiguousarray(points1[0:2,:].T)
    points2 = np.ascontiguousarray(points2[0:2,:].T)
    E, inliers = cv.findEssentialMat(points1, points2, camera_intrinsics.K)
    inliers = np.squeeze(np.array(inliers, dtype=np.bool))

    return E, inliers

def estimate_essential_matrix_ransac(points1 : NDArray[Shape["3,Any"], Float],
                                     points2 : NDArray[Shape["3,Any"], Float],
                                     num_iters : int = 1000,
                                     threshold : float = 0.01) -> Tuple[int, NDArray[Shape["3,3"], Float], list[int]]:
    
    if points1.shape[1] != points2.shape[1]:
        raise Exception("Point arrays must have the same number of points!")
    
    num_points = points1.shape[1]
    num_points_for_model = 8
    
    best_score = 0
    best_E = np.zeros((3,3))
    best_inlier_indices = []
    for _ in range(num_iters):
        rand_indices = np.random.choice(num_points, num_points_for_model, replace=False)
        E = get_essential_matrix(points1[:,rand_indices], points2[:,rand_indices])
        d = get_coplainarity_distances(points1, points2, E)
        inlier_indices = d < threshold
        score = np.sum(inlier_indices)
        
        if score > best_score:
            best_score = score
            best_E = E
            best_inlier_indices = inlier_indices
                
                
    return best_score, best_E, best_inlier_indices

def estimate_fundamental_matrix_ransac(points1 : NDArray[Shape["3,Any"], Float],
                                       points2 : NDArray[Shape["3,Any"], Float],
                                       num_iters : int = 1000,
                                       threshold : float = 0.01) -> Tuple[int, NDArray[Shape["3,3"], Float], list[int]]:
    

    if points1.shape[1] != points2.shape[1]:
        raise Exception("Point arrays must have the same number of points!")
    
    num_points = points1.shape[1]
    num_points_for_model = 8
    
    best_score = 0
    best_F = np.zeros((3,3))
    best_inlier_indices = []
    for _ in range(num_iters):
        rand_indices = np.random.choice(num_points, num_points_for_model, replace=False)
        F = get_fundamental_matrix(points1[:,rand_indices], points2[:,rand_indices])
        
        d = get_coplainarity_distances(points1, points2, F)
        inlier_indices = d < threshold
        score = np.sum(inlier_indices)
        
        if score > best_score:
            best_score = score
            best_F = F
            best_inlier_indices = inlier_indices
                
                
    return best_score, best_F, best_inlier_indices
        
def get_essential_matrix(points1 : list[NDArray[Shape["3,Any"], Float]], 
                           points2 : list[NDArray[Shape["3,Any"], Float]]) -> NDArray[Shape["3,3"], Float]:
    
    num_allowed_points = 8
    if points1.shape[1] != num_allowed_points or points2.shape[1] != num_allowed_points:
        raise Exception("Need 8 points each for 8-point algorithm!")
    
    # normalize points 
    norm_points1, trans1 = zero_center_normalize_homo_points(points1)
    norm_points2, trans2 = zero_center_normalize_homo_points(points2)
    
    m = np.zeros((8,9))
    for i in range(num_allowed_points):
        m[i,:] = (np.kron(norm_points1[:,i], norm_points2[:,i]))
    
    # get null space
    u,s,v = np.linalg.svd(m)
    E = v[-1,:].reshape((3,3))

    # enforce rank-2 constraint and same eigenvalues
    uE,sE,vE = np.linalg.svd(E)
    sE[-1] = 0
    sE[0] = 1
    sE[1] = 1
    E = uE @ np.diag(sE) @ vE

    E = trans2.T @ E @ trans1
    
    if E[-1,-1] < 0 :
        E = -E
    
    return E

def get_fundamental_matrix(points1 : list[NDArray[Shape["3,Any"], Float]], 
                           points2 : list[NDArray[Shape["3,Any"], Float]]) -> NDArray[Shape["3,3"], Float]:
    
    num_allowed_points = 8
    if points1.shape[1] != num_allowed_points or points2.shape[1] != num_allowed_points:
        raise Exception("Need 8 points each for 8-point algorithm!")
    
    # normalize points 
    norm_points1, trans1 = zero_center_normalize_homo_points(points1)
    norm_points2, trans2 = zero_center_normalize_homo_points(points2)
    
    m = np.zeros((8,9))
    for i in range(num_allowed_points):
        m[i,:] = (np.kron(norm_points1[:,i], norm_points2[:,i]))
    
    # get null space
    u,s,v = np.linalg.svd(m)
    F = v[-1,:].reshape((3,3))
    
    # enforce rank-2 constraint
    uF, sF, vF = np.linalg.svd(F)
    sF[-1] = 0
    F = uF @ np.diag(sF) @ vF
    
    F = trans2.T @ F @ trans1

    # normalize F
    F = F / np.linalg.norm(F,2)
    
    if F[-1,-1] < 0 :
        F = -F
    
    return F

def get_essential_matrix_from_fundamental_matrix(fundamental_matrix : NDArray[Shape["3,3"], Float],
                                                 camera_intrinsics1: CameraIntrinsics,
                                                 camera_intrinsics2: CameraIntrinsics) -> NDArray[Shape["3,3"], Float]:

    E = camera_intrinsics2.K.T @ fundamental_matrix @ camera_intrinsics1.K

    # enforce rank 2 and normalize singular values
    u,s,v = np.linalg.svd(E)
    s[-1] = 0
    s_average = (s[0] + s[1])/2
    s[0] = s_average
    s[1] = s_average
    E = u @ np.diag(s) @ v

    return E

def decompose_essential_matrix(essential_matrix : NDArray[Shape["3,3"], Float]) -> list[NDArray[Shape["3,3"], Float]]:

    u, s, v = np.linalg.svd(essential_matrix)
    # W and Z chosen such as ZW is equal to s from E = usv'
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])

    possible_rotations = []
    rot1 =  u @ W @ v
    if np.linalg.det(rot1) < 0:
        rot1 = -rot1
    rot2 = u @ W.T @ v
    if np.linalg.det(rot2) < 0:
        rot2 = -rot2
    
    possible_rotations.append(rot1)
    possible_rotations.append(rot2)

    possible_translations = []
    skew1 = u @ Z @ u.T
    tran1 = np.expand_dims(np.array([skew1[2,1], skew1[0,2], skew1[1,0]]), axis=1)
    possible_translations.append(tran1)
    possible_translations.append(-tran1)

    possible_camera_extrinsics = []
    for rot in possible_rotations:
        for trans in possible_translations:
            possible_camera_extrinsics.append(CameraExtrinsics(trans, rot))

    return possible_camera_extrinsics

def get_realizable_relpose_from_essential_matrix(
    points1 : list[NDArray[Shape["3,Any"], Float]], 
    points2 : list[NDArray[Shape["3,Any"], Float]],
    camera_intrinsics1 : CameraIntrinsics,
    camera_intrinsics2 : CameraIntrinsics,
    essential_matrix : NDArray[Shape["3,3"], Float]
    ) -> Tuple[NDArray[Shape["3,1"], Float], NDArray[Shape["3,3"], Float], Float]:

    # Get possible cameras
    possible_cameras1 = []
    possible_cameras2 = []
    no_translation = np.zeros((3,1))
    no_rotation = np.diag(np.ones((3,)))
    zero_camera_extrinsic = CameraExtrinsics(no_translation, no_rotation)
    possible_camera_extrinsics = decompose_essential_matrix(essential_matrix)
    for possible_camera_extrinsic in possible_camera_extrinsics:
        possible_cameras1.append(Camera(camera_intrinsics1, zero_camera_extrinsic))
        possible_cameras2.append(Camera(camera_intrinsics2, possible_camera_extrinsic))

    # triangulate points with possible cameras and check if points are in front of cameras
    num_points_behind_either_camera = []
    for i in range(len(possible_cameras1)):
        world_points_wrt_camera1, _, _ = triangulate_world_points(points1, 
                                                      points2, 
                                                      possible_cameras1[i], 
                                                      possible_cameras2[i],
                                                      "LLS",
                                                      0.5)
        world_points_wrt_camera1 = world_points_wrt_camera1
        world_points_wrt_camera2 = possible_cameras2[i].rotation @ (world_points_wrt_camera1 - possible_cameras2[i].translation)
        num_points_behind_either_camera.append(np.sum(world_points_wrt_camera1[2,:] < 0 | (world_points_wrt_camera2[2,:] < 0)))

    best_index = np.argmin(num_points_behind_either_camera)
    best_rel_translation = possible_cameras2[best_index].translation
    best_rel_rotation = normalize(possible_cameras2[best_index].rotation, axis=1, norm='l2')

    return best_rel_translation, best_rel_rotation
    
def get_coplainarity_distances(points1 : NDArray[Shape["3,Any"], Float],
                              points2 : NDArray[Shape["3,Any"], Float],
                              F : NDArray[Shape["3,3"], Float]) -> NDArray[Shape["Any,1"], Float]:
    
    return np.power(np.sum(np.multiply((points2.T @ F).T, points1), axis=0), 2)

def zero_center_normalize_homo_points(homo_points : NDArray[Shape["Any,Any"], Float]) -> Tuple[NDArray[Shape["Any,1"], Float], NDArray[Shape["Any,Any"], Float]]:
    
    # extract points from homogeneous point
    points = homo_points[:-1]

    # make sure point is zero mean, and average coordinate distance from mean is sqrt(2)
    means = np.mean(points, axis=1)
    points = points - np.expand_dims(means, axis=1)
    mean_distance_from_center = np.mean(np.sqrt(np.sum(np.power(points,2),axis=0)))
    scale =  math.sqrt(2) / mean_distance_from_center
    points = points * scale
    norm_points = np.append(points, np.ones((1,points.shape[1])), axis=0)

    # form transformation matrix for the normalization
    transformation = np.diag(np.full((points.shape[0]), scale))
    transformation = np.append(transformation, -1*scale*np.expand_dims(means, axis=1), axis=1)
    transformation = np.append(transformation, np.zeros((1, transformation.shape[1])), axis=0)
    transformation[-1,-1] = 1
    
    return norm_points, transformation


