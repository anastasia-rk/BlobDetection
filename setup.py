# Imports
import cv2
import numpy as np
from IPython.display import display, Image
from ipywidgets import *
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.util import img_as_ubyte
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, denoise_tv_bregman, estimate_sigma)
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

def conservative_smoothing_gray(data, filterSize):
    temp = []
    indexer = filterSize // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in tqdm(range(nrow)):
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
    im = cv2.imread("InputPhotos/IMG_7479.jpg", cv2.IMREAD_GRAYSCALE)
    # Img = cv2.imread(‘./ img.jpg’, cv2.IMREAD_GRAYSCALE)

    # denoise the image before increasing contrast
    filterSize = 11
    sigmaSpace = 150
    im_mean_blur = cv2.blur(im, (filterSize, filterSize))
    im_med_blur = cv2.medianBlur(im, filterSize)
    im_bilat = cv2.bilateralFilter(im, d=filterSize*2, sigmaColor=10, sigmaSpace=sigmaSpace)
    check_blur1 = np.hstack((im, im_mean_blur))  # stacking images side-by-side
    check_blur2 = np.hstack((im_med_blur, im_bilat))
    check_blur = np.vstack((check_blur1, check_blur2))
    cv2.imshow('check_blur.png', check_blur)

    # denoise with skimage filters - this is to removed the small scale grain in the images that may be picked up by the detector
    # there are several methods of doing skimage denoising
    im_bergman = img_as_ubyte(denoise_tv_bregman(im, weight=1.2, channel_axis=None))
    im_wavelet = img_as_ubyte(denoise_wavelet(im, rescale_sigma=True, mode='hard', method='BayesShrink', sigma=15, channel_axis=None))
    im_chambolle = img_as_ubyte(denoise_tv_chambolle(im, weight=0.8, channel_axis=None))
    im_cons_blur = conservative_smoothing_gray(im, 19)

    cv2.imshow('conservative.png', im_cons_blur)

    check_blur3 = np.hstack((im_bergman, im_cons_blur))  # stacking images side-by-side
    check_blur4 = np.hstack((im_chambolle, im_wavelet))
    check_blur_skimage = np.vstack((check_blur3, check_blur4))
    cv2.imshow('check_blur_skimage.png', check_blur_skimage)
    plt.savefig('output.png')

    # Increase contrast - this is useful to do for overexposed images, but may be dropped if it does not improve the final count
    # create a CLAHE object (Arguments are optional).
    tileSize = 70
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(tileSize, tileSize))
    im_cl_med = clahe.apply(im_med_blur)
    im_cl_ber = clahe.apply(im_bergman)
    im_cl_cons = clahe.apply(im_cons_blur)
    compare_hists = np.hstack((im_cl_med, im_cl_ber, im_cl_cons))  # stacking images side-by-side
    cv2.imshow('compare_hists.png', compare_hists)

    # Set up the detector with default parameters:
    detector = cv2.SimpleBlobDetector()
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 1000

    # params.filterByColor = True
    # params.blobColor=0
    # params.blobColor = 255

    # the most basic way to detect the blobs is by limiting the hue of the greyscale image.
    # note that the colors run from 0 = white to 255 = black
    params.minThreshold = 0
    params.maxThreshold = 50

    # Filter by Area. In this case, there is no clear upper or lower limit because the area is defined in pixels and will depend strongly on your image resolution
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 2500
    #
    # # Filter by Circularity: 1 - perfect circle, 0 - long ellipse
    params.filterByCircularity = True
    params.minCircularity = 0.5
    #
    # # Filter by Convexity 1 - perfectly convex, 0 - allows for many concave areas
    params.filterByConvexity = True
    params.minConvexity = 0.8
    #
    # # Filter by Inertia - how long is your ellipse allowed to be
    params.filterByInertia = False
    params.minInertiaRatio = 0.5
    params.minInertiaRatio = 0.99

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)


    # Detect blobs from the smoothed images

    keypoints1 = detector.detect(im)
    keypoints2 = detector.detect(im_med_blur)
    keypoints3 = detector.detect(im_cl_ber)
    keypoints4 = detector.detect(im_cl_cons)

    # Draw detected blobs as green circles.
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # ensures the size of the circle corresponds to the size of blob
    blank = np.zeros((1, 1)) #    blank = np.zeros((1, 1)) #
    im_with_keypoints1 = cv2.drawKeypoints(im, keypoints1, blank, (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints2 = cv2.drawKeypoints(im_cl_med, keypoints2, blank, (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints3 = cv2.drawKeypoints(im_cl_ber, keypoints3, blank, (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints4 = cv2.drawKeypoints(im_cl_cons, keypoints4, blank, (0, 255, 0),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(len(keypoints1))
    print(len(keypoints2))
    print(len(keypoints3))
    print(len(keypoints4))
    # Show keypoints
    compare_blobs1 = np.hstack((im_with_keypoints1, im_with_keypoints2))
    compare_blobs2 = np.hstack((im_with_keypoints3, im_with_keypoints4))
    compare_blobs = np.vstack((compare_blobs1,compare_blobs2))
    cv2.imshow("Keypoints", compare_blobs)
    cv2.waitKey(0)
    print('pause here')
# Send main
