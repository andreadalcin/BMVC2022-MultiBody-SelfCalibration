import sys

import cv2 as cv
import numpy as np
import scipy.io

from b_tlinkage import OUTLIER_THRESHOLD
from c_get_preference_matrix_fm import get_preference_matrix_fm
from d_clustering import clustering
from utils_t_linkage import show_pref_matrix, get_cluster_mask

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


def extract_and_match():
    numCams = 7
    numMotions = 3

    fileName = f'matches/mathces-{numCams}-{numMotions}.mat'
    print(fileName)

    mat = scipy.io.loadmat(fileName)
    matches = mat['matches']
    numF = matches.shape[0]

    Fs = np.zeros((3, 3, numF * numMotions))

    for f in range(numF):
        src_pts = matches[f, :, 0:2]
        dst_pts = matches[f, :, 2:4]

        num_of_points = src_pts.shape[0]

        kp_src = [cv.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1) for p
                  in src_pts]
        kp_dst = [cv.KeyPoint(x=p[0], y=p[1], _size=1.0, _angle=1.0, _response=1.0, _octave=1, _class_id=-1) for p
                  in dst_pts]
        good_matches = [cv.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=1) for i in range(len(src_pts))]

        # region Get preference matrix
        tau = 14
        pref_m = get_preference_matrix_fm(kp_src, kp_dst, good_matches, tau)
        show_pref_matrix(pref_m, 'label')
        # endregion

        # region Clustering
        clusters, pref_m = clustering(pref_m)
        clusters_mask = get_cluster_mask(clusters, num_of_points, OUTLIER_THRESHOLD)
        # endregion

        unique_elements, frequency = np.unique(clusters_mask, return_counts=True)
        sorted_indexes = np.argsort(frequency)[::-1]
        sorted_by_freq = unique_elements[sorted_indexes]
        sorted_by_freq = np.delete(sorted_by_freq, sorted_by_freq == 0)

        # -- Find first F
        F_1, mask_1 = cv.findFundamentalMat(src_pts[clusters_mask == sorted_by_freq[0]],
                                            dst_pts[clusters_mask == sorted_by_freq[0]],
                                            cv.FM_LMEDS)

        F_2, mask_2 = cv.findFundamentalMat(src_pts[clusters_mask == sorted_by_freq[1]],
                                            dst_pts[clusters_mask == sorted_by_freq[1]],
                                            cv.FM_LMEDS)

        F_3, mask_3 = cv.findFundamentalMat(src_pts[clusters_mask == sorted_by_freq[2]],
                                            dst_pts[clusters_mask == sorted_by_freq[2]],
                                            cv.FM_LMEDS)

        Fs[:, :, f * numMotions] = F_1
        Fs[:, :, f * numMotions + 1] = F_2
        Fs[:, :, f * numMotions + 2] = F_3

    scipy.io.savemat(
        f"output/test-{numCams}-{numMotions}.mat",
        {'Fs': Fs})


if __name__ == "__main__":
    extract_and_match()
