# Imports
import cv2
import numpy as np
import scipy as sp
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# # for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
def conservative_smoothing_gray(data, filterSize):
    temp = []
    indexer = filterSize // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i - indexer, i + indexer + 1):
                for m in range(j - indexer, j + indexer + 1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k, m])
            temp.remove(data[i, j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i, j] > max_value:
                new_image[i, j] = max_value
            elif data[i, j] < min_value:
                new_image[i, j] = min_value
            temp = []
    return new_image.copy()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read image
    im = cv2.imread("InputPhotos/4.jpg", cv2.IMREAD_GRAYSCALE)
    # Img = cv2.imread(‘./ img.jpg’, cv2.IMREAD_GRAYSCALE)

    # denoise the image before increasing contrast
    filterSize = 5
    im_mean_blur = cv2.blur(im, (filterSize, filterSize))
    im_med_blur = cv2.medianBlur(im, filterSize)
    # im_cons_blur = conservative_smoothing_gray(im, 25)
    check_blur1 = np.hstack((im, im_mean_blur))  # stacking images side-by-side
    check_blur2 = np.hstack((im_med_blur, im_med_blur))
    check_blur = np.vstack((check_blur1, check_blur2))
    cv2.imshow('check_blur.png', check_blur)

    im_contrast = cv2.equalizeHist(im_med_blur)
    check_hist = np.hstack((im, im_med_blur, im_contrast))  # stacking images side-by-side
    cv2.imshow('check_hist.png', check_hist)
    # compare contrast enhancement methods

    # create a CLAHE object (Arguments are optional).
    tileSize = 25
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(tileSize, tileSize))
    im_cl = clahe.apply(im_med_blur)
    compare_hists = np.hstack((im, im_med_blur, im_cl))  # stacking images side-by-side
    cv2.imshow('check_clahe.png', compare_hists)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 500

    params.filterByColor = True
    params.blobColor = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 80
    params.maxArea = 500
    #
    # # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3
    #
    # # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.4
    #
    # # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.51
    params.minInertiaRatio = 0.99

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs.

    keypoints1 = detector.detect(im)
    keypoints2 = detector.detect(im_contrast)
    keypoints3 = detector.detect(im_cl)

    # Draw detected blobs as red circles.
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # ensures the size of the circle corresponds to the size of blob
    blank = np.zeros((1, 1)) #    blank = np.zeros((1, 1)) #
    im_with_keypoints1 = cv2.drawKeypoints(im, keypoints1, blank, (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints2 = cv2.drawKeypoints(im_contrast, keypoints2, blank, (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints3 = cv2.drawKeypoints(im_cl, keypoints3, blank, (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    compare_blobs = np.hstack((im_with_keypoints1, im_with_keypoints2, im_with_keypoints3))
    cv2.imshow("Keypoints", compare_blobs)
    cv2.waitKey(0)

# Send main
