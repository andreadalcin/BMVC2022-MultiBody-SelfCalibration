import cv2 as cv
import numpy as np

from c_get_preference_matrix_fm import get_preference_matrix_fm
from c_get_preference_matrix_h import get_preference_matrix_h
from d_clustering import clustering
from utils_t_linkage import plotMatches, show_pref_matrix, plot_clusters, get_cluster_mask

OUTLIER_THRESHOLD = 12


def t_linkage(tau, label_k, mode):

    # region Get image path from label_k
    img_i = "../resources/adel" + mode + "_imgs/" + label_k + "1.png"
    img_j = "../resources/adel" + mode + "_imgs/" + label_k + "2.png"
    # endregion

    ####################################################################################################################
    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints1, descriptors1 = detector.detectAndCompute(cv.cvtColor(cv.imread(img_i), cv.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = detector.detectAndCompute(cv.cvtColor(cv.imread(img_j), cv.COLOR_BGR2GRAY), None)

    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # -- Output to .mat
    src_pts = np.array([keypoints1[mat.queryIdx].pt for mat in good_matches])
    dst_pts = np.array([keypoints2[mat.trainIdx].pt for mat in good_matches])


    # region Load data points
    # data_dict = loadmat("../resources/adel" + mode + "/" + label_k + ".mat")
    # points = data_dict['data']
    # points = np.transpose(points)
    # src_pts = points[:, 0:2]
    # dst_pts = points[:, 3:5]
    num_of_points = src_pts.shape[0]
    # endregion

    # region Sort points so to graphically emphasize the structures in the preference matrix
    # label = data_dict['label'].flatten()
    # idx = np.argsort(label)

    # src_pts = np.array([src_pts[i] for i in idx])
    # dst_pts = np.array([dst_pts[i] for i in idx])
    # clusters_mask_gt = np.array([label[i] for i in idx])
    # endregion

    # region Build kp and good_matches
    kp_src = [cv.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1) for p in src_pts]
    kp_dst = [cv.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1) for p in dst_pts]
    good_matches = [cv.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=1) for i in range(len(src_pts))]
    plotMatches(img_i, img_j, kp_src, kp_dst, good_matches)
    # endregion

    # | ############################################################################################################# |

    # region Get preference matrix
    if mode == "FM":
        pref_m = get_preference_matrix_fm(kp_src, kp_dst, good_matches, tau)
    else:  # mode == "H"
        pref_m = get_preference_matrix_h(kp_src, kp_dst, good_matches, tau)

    show_pref_matrix(pref_m, label_k)
    # endregion

    # region Clustering
    clusters, pref_m = clustering(pref_m)
    clusters_mask = get_cluster_mask(clusters, num_of_points, OUTLIER_THRESHOLD)
    # endregion

    # region Plot clusters
    # plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask_gt, label_k + " - Ground-truth")
    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label_k + " - Estimation")
    # endregion

    # region Compute Misclassification Error
    # err, num_of_pts = compute_errors(clusters_mask, clusters_mask_gt)
    # me = err / num_of_pts  # compute misclassification error
    # print("ME % = " + str(round(float(me), 4)))
    # endregion


