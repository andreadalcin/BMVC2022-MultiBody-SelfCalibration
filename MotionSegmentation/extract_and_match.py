import sys
from os import listdir
from os.path import isfile, join

import cv2 as cv
import numpy as np
import scipy.io

sys.path.append('./src')


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


def extract_and_match(label, img1_file_name, img2_file_name, display_F=True):
    print(img1_file_name)
    print(img2_file_name)

    img1 = cv.imread(f'{label}/{img1_file_name}')
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    img2 = cv.imread(f'{label}/{img2_file_name}')
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    ####################################################################################################################
    # -- Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints1, descriptors1 = detector.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2_gray, None)

    # -- Matching descriptor vectors with a FLANN based matcher
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
    X0 = np.array([keypoints1[mat.queryIdx].pt for mat in good_matches])
    Y0 = np.array([keypoints2[mat.trainIdx].pt for mat in good_matches])

    ####################################################################################################################
    # -- Find first F
    F_1, mask_1 = cv.findFundamentalMat(X0, Y0, cv.FM_RANSAC)
    print(f'number of inliers 1: {np.shape(mask_1)[0]}')

    if display_F:
        display = draw_match(img1, img2, X0[np.squeeze(mask_1 == 1)], Y0[np.squeeze(mask_1 == 1)])
        cv.imshow("fundamental matrix estimation visualization (1)", resize(display, scale_percent=20))
        print('please press any key to terminate window')
        k = cv.waitKey(0)
        cv.destroyAllWindows()

    return {'F_1': F_1, 'matches_1': mask_1.shape[0]}


if __name__ == "__main__":
    label = "<<dataset>>>"
    images_path = f"{label}/"

    file_names = sorted([f for f in listdir(images_path) if ".JPG" in f and isfile(join(images_path, f))])
    print(file_names)

    num_images = len(file_names)

    Fs = np.zeros((3, 3, num_images, num_images, 1))
    matches_1 = np.zeros((num_images, num_images))

    for i in range(num_images - 1):
        for j in range(i + 1, num_images):
            print(f'i: {i}, j: {j}')
            out = extract_and_match(label, file_names[i], file_names[j])
            Fs[:, :, i, j, 0] = out['F_1']
            matches_1[i, j] = out['matches_1']

    scipy.io.savemat(f"{label}/{label}.mat", {'Fs': Fs, 'matches': matches_1})
