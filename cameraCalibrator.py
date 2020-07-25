import glob
import os

import cv2
import numpy as np

from utils import (
    outerContour
)

warpedW = 700
warpedH = 900


def checkBlankArea(warped):
    """
    Check the mean of the area expected to be an empty chess.
    To align the chessboard image find a minimum of this value.
    """
    roi = warped[75:160, 510:635]
    mean = cv2.mean(roi)
    return mean[0]


def cropRect(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    minContourLength = 10
    polys = []
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) >= minContourLength:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            curve = cv2.approxPolyDP(contour, epsilon, True)
            if len(curve) == 4 and cv2.isContourConvex(curve):
                polys.append(curve)

    polys.sort(key=lambda x: outerContour(x, thresh), reverse=False)
    biggerContour = polys[0]

    destPoints: np.ndarray = np.array([[[0, warpedH]], [[0, 0]], [[warpedW, 0]], [[warpedW, warpedH]]])
    M = cv2.findHomography(biggerContour, destPoints)[0]
    warped = cv2.warpPerspective(gray, M, (warpedW, warpedH))

    currMax = checkBlankArea(warped)
    for i in range(3):
        biggerContour = np.roll(biggerContour, shift=1, axis=0)
        M2 = cv2.findHomography(biggerContour, destPoints)[0]
        rotated = cv2.warpPerspective(gray, M2, (warpedW, warpedH))
        rotatedScore = checkBlankArea(rotated)
        if rotatedScore > currMax:
            M = M2
            warped = rotated
            currMax = rotatedScore

    return M, warped


def genChessboardCorners():
    width = 160
    height = 200
    # borderPoints = [[10, 10], [150, 10], [150, 190], [10, 190]]
    borderPoints = [[10, 10], [150, 10], [150, 190]]
    innerPoints = \
        [[20, j] for j in range(40, 180, 20)] + \
        [[i, j] for i in range(40, 120, 20) for j in range(20, 200, 20)] + \
        [[120, j] for j in range(40, 200, 20)] + \
        [[140, j] for j in range(60, 180, 20)]

    points = borderPoints + innerPoints
    return np.array([[[x / width * warpedW, y / height * warpedH]] for x, y in points]).astype(np.float32)


def findChessboardCorners(gray):
    M, cropped = cropRect(gray)
    # cv2.imshow("cropped", cropped)
    kernel = np.ones((6, 6), np.uint8)
    open = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("open", open)
    # cv2.imshow("close", close)

    invM = cv2.invert(M)[1]
    targetCorners = genChessboardCorners()
    imagePoints = []
    objectPoints = []

    winSize = (32, 32)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 200, 0.1)
    corners = cv2.cornerSubPix(close, targetCorners, winSize, zeroZone, criteria)
    # corners = targetCorners
    projected = cv2.perspectiveTransform(corners, invM)

    for t, p in zip(targetCorners, projected):
        tx, ty = t[0]
        px, py = p[0]
        objectPoints.append([tx, ty, 0])
        imagePoints.append([px, py])

    return imagePoints, objectPoints


def run():
    imagePoints = []
    objectPoints = []

    os.chdir("./project_data/G3DCV2020_data_part1_calibration/calib/")
    i = 0
    for file in glob.glob("*.png"):
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        imgImagePoints, imgObjectPoints = findChessboardCorners(gray)

        imagePoints.append(imgImagePoints)
        objectPoints.append(imgObjectPoints)

        for c in imgImagePoints:
            px, py = c
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
        cv2.imshow(file, img)
        cv2.waitKey(1)

        print(f"Image {i}/50")
        i += 1

    imagePoints = np.array(imagePoints).astype(np.float32)
    imagePoints = np.reshape(
        imagePoints, (imagePoints.shape[0], imagePoints.shape[1], 1, 2)
    )
    objectPoints = np.array(objectPoints).astype(np.float32)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints, imagePoints, (1296, 972), None, None
    )
    print(f"\n\nRMS: {ret}")
    print(f"\n\nK: {K}")
    print(f"Distortion parameters:\n{dist}")
    print(f"Images used for calibration: {imagePoints.shape[0]}/50")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


run()
