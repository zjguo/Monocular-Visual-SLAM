import cv2 as cv
import numpy as np

def detect_and_match_pair(
    img1, 
    img2, 
    n_features = 200,
    scale_factor = 1.2,
    n_levels=15,
    fast_threshold=31,
    score_type=0,
    detector_type='orb',
    lowes_ratio = 0.75):

    # get keypoints
    kpts1 = []
    kpts2 = []
    orb = cv.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels, fastThreshold=fast_threshold, scoreType=score_type)
    if detector_type == 'orb':
        kpts1 = orb.detect(img1, None)
        kpts2 = orb.detect(img2, None)
    elif detector_type == 'gftt':
        locs1 = cv.goodFeaturesToTrack(np.mean(img1, axis=2).astype(np.uint8), n_features, qualityLevel=0.01, minDistance=7)
        kpts1 = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in locs1]
        locs2 = cv.goodFeaturesToTrack(np.mean(img2, axis=2).astype(np.uint8), n_features, qualityLevel=0.01, minDistance=7)
        kpts2 = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in locs2]
    
    # find the descriptors with ORB
    kpts1, des1 = orb.compute(img1,kpts1)
    kpts2, des2 = orb.compute(img2,kpts2)

    # lowes ratio test
    good_matches = []
    if lowes_ratio >= 1.0:
        # Match descriptors.
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good_matches = matches
    else:
        # Match descriptors.
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        for m,n in matches:
            if m.distance < lowes_ratio*n.distance:
                good_matches.append(m)

    good_matches = sorted(good_matches, key = lambda x:x.distance)

    return kpts1 ,kpts2, des1, des2, good_matches

def get_matched_point_matrix(kp, matches_indices):

    points = np.array([kp[i].pt for i in range(len(kp))])
    points = np.concatenate((points.T, np.ones((1,points.shape[0]))), axis=0)
    matched_points = np.array([points[:,i] for i in matches_indices]).T

    return matched_points

def show_matches(img, kpts1, kpts2, matches, mask):
    filtered_kpts2 = [kpts2[i.trainIdx] for i in matches]
    filtered_kpts2 = [filtered_kpts2[i] for i,x in enumerate(mask) if x == True]
    filtered_kpts1 = [kpts1[i.queryIdx] for i in matches]
    filtered_kpts1 = [filtered_kpts1[i] for i,x in enumerate(mask) if x == True]
    output_image = cv.drawKeypoints(img, filtered_kpts2, 0, (0, 0, 255),
                            flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    output_image = cv.drawKeypoints(output_image, filtered_kpts1, 0, (0, 0, 255),
                            flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    for i in range(len(filtered_kpts1)):
        output_image = cv.line(output_image, tuple(int(x) for x in filtered_kpts1[i].pt), tuple(int(x) for x in filtered_kpts2[i].pt), (255, 0, 0), 2)
    output_image = cv.resize(output_image, (640,360))
    cv.imshow("current frame", output_image)
    cv.waitKey(25)

