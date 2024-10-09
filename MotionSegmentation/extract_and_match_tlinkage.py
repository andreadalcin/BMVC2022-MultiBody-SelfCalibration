import sys
from os import listdir
from os.path import isfile, join

import cv2 as cv
import numpy as np
import scipy.io

from b_tlinkage import OUTLIER_THRESHOLD
from c_get_preference_matrix_fm import get_preference_matrix_fm
from d_clustering import clustering
from utils_t_linkage import show_pref_matrix, plotMatches, get_cluster_mask, plot_clusters

sys.path.append('./src')
import RANSAC


def resize(img, scale_percent=50):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)


def draw_match(img1, img2, corr1, corr2):
    corr1 = [cv.KeyPoint(corr1[i, 0], corr1[i, 1], 1) for i in range(corr1.shape[0])]
    corr2 = [cv.KeyPoint(corr2[i, 0], corr2[i, 1], 1) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv.DMatch(i, i, 0) for i in range(len(corr1))]

    return cv.drawMatches(img1, corr1, img2, corr2, draw_matches, None,
                          matchColor=(0, 255, 0),
                          singlePointColor=(0, 0, 255),
                          flags=0
                          )


def extract_and_match(label, img_i, img_j, display_F=False):
    print(img_i)
    print(img_j)

    img_i = f'{label}/{img_i}'
    img_j = f'{label}/{img_j}'

    img1 = cv.imread(img_i)
    img2 = cv.imread(img_j)

    ####################################################################################################################
    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints1, descriptors1 = detector.detectAndCompute(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = detector.detectAndCompute(cv.cvtColor(img2, cv.COLOR_BGR2GRAY), None)

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

    # t-linkage
    src_pts = np.array([keypoints1[mat.queryIdx].pt for mat in good_matches])
    dst_pts = np.array([keypoints2[mat.trainIdx].pt for mat in good_matches])

    num_of_points = src_pts.shape[0]

    kp_src = [cv.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1) for p in
              src_pts]
    kp_dst = [cv.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1) for p in
              dst_pts]
    good_matches = [cv.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=1) for i in range(len(src_pts))]
    plotMatches(img_i, img_j, kp_src, kp_dst, good_matches)

    # | ############################################################################################################# |

    # region t-linkage
    tau = 14
    pref_m = get_preference_matrix_fm(kp_src, kp_dst, good_matches, tau)
    show_pref_matrix(pref_m, label)

    clusters, pref_m = clustering(pref_m)
    clusters_mask = get_cluster_mask(clusters, num_of_points, OUTLIER_THRESHOLD)

    plot_clusters(img_i, img_j, src_pts, dst_pts, clusters_mask, label + " - Estimation")

    unique_elements, frequency = np.unique(clusters_mask, return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    sorted_by_freq = unique_elements[sorted_indexes]
    sorted_by_freq = np.delete(sorted_by_freq, sorted_by_freq == 0)
    # endregion

    ####################################################################################################################
    maxTrials = 5000  # maximum number of trials
    th = 2.0  # inlier-outlier threshold in pixels
    confidence = 0.999999  # confidence value
    DEGEN = True  # degeneracy updating
    LO = True  # local optimization

    # -- Find first F
    F_1, mask_1 = RANSAC.findFundamentalMatrix(src_pts[clusters_mask == sorted_by_freq[0]],
                                               dst_pts[clusters_mask == sorted_by_freq[0]],
                                               maxTrials,
                                               th,
                                               confidence,
                                               DEGEN,
                                               LO)

    F_2, mask_2 = RANSAC.findFundamentalMatrix(src_pts[clusters_mask == sorted_by_freq[1]],
                                               dst_pts[clusters_mask == sorted_by_freq[1]],
                                               maxTrials,
                                               th,
                                               confidence,
                                               DEGEN,
                                               LO)

    # -- Find third F
    F_3, mask_3 = RANSAC.findFundamentalMatrix(src_pts[clusters_mask == sorted_by_freq[2]],
                                               dst_pts[clusters_mask == sorted_by_freq[2]],
                                               maxTrials,
                                               th,
                                               confidence,
                                               DEGEN,
                                               LO)

    if display_F:
        cluster_src_pts = src_pts[clusters_mask == sorted_by_freq[0], :]
        cluster_dst_pts = dst_pts[clusters_mask == sorted_by_freq[0], :]
        display = draw_match(img1, img2, cluster_src_pts[mask_1, :], cluster_dst_pts[mask_1, :])
        cv.imshow("fundamental matrix estimation visualization (1)", resize(display, scale_percent=20))
        print('please press any key to terminate window')
        k = cv.waitKey(0)
        cv.destroyAllWindows()

        cluster_src_pts = src_pts[clusters_mask == sorted_by_freq[1], :]
        cluster_dst_pts = dst_pts[clusters_mask == sorted_by_freq[1], :]
        display = draw_match(img1, img2, cluster_src_pts[mask_2, :], cluster_dst_pts[mask_2, :])
        cv.imshow("fundamental matrix estimation visualization (2)", resize(display, scale_percent=20))
        print('please press any key to terminate window')
        k = cv.waitKey(0)
        cv.destroyAllWindows()

        cluster_src_pts = src_pts[clusters_mask == sorted_by_freq[2], :]
        cluster_dst_pts = dst_pts[clusters_mask == sorted_by_freq[2], :]
        display = draw_match(img1, img2, cluster_src_pts[mask_3, :], cluster_dst_pts[mask_3, :])
        cv.imshow("fundamental matrix estimation visualization (3)", resize(display, scale_percent=20))
        print('please press any key to terminate window')
        k = cv.waitKey(0)
        cv.destroyAllWindows()

    return {'F1': F_1, 'F2': F_2, 'F_3': F_3, 'matches_1': 0, 'matches_2': 0, 'matches_3': 0}


if __name__ == "__main__":
    label = "<<dataset>>>"
    images_path = f'{label}/'

    file_names = sorted([f for f in listdir(images_path) if ".JPG" in f and isfile(join(images_path, f))])
    print(file_names)

    num_images = len(file_names)

    Fs = np.zeros((3, 3, num_images, num_images, 3))
    matches_1 = np.zeros((num_images, num_images))
    matches_2 = np.zeros((num_images, num_images))
    matches_3 = np.zeros((num_images, num_images))

    for i in range(num_images - 1):
        for j in range(i + 1, num_images):
            print(f'i: {i}, j: {j}')
            out = extract_and_match(label, file_names[i], file_names[j])
            Fs[:, :, i, j, 0] = out['F_1']
            matches_1[i, j] = out['matches_1']
            Fs[:, :, i, j, 1] = out['F_2']
            matches_2[i, j] = out['matches_2']
            Fs[:, :, i, j, 2] = out['F_3']
            matches_3[i, j] = out['matches_3']

    scipy.io.savemat(f"{label}/{label}_tlinkage.mat",
                     {'Fs': Fs, 'matches_1': matches_1, 'matches_2': matches_2, 'matches_3': matches_3})
