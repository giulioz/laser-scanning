import glob
import os

import cv2
import numpy as np

warpedW = 700
warpedH = 900


def outerContour(contour, gray):
    margin = 10
    kernel = np.ones((margin, margin), np.uint8)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    eroded = cv2.erode(mask, kernel)
    mask = cv2.bitwise_xor(eroded, mask)
    mean = cv2.mean(gray, mask)
    return mean[0]


def checkBlankArea(warped):
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


def findChessboardCorners(cropped):
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 9
    gradientSize = 9
    useHarrisDetector = False
    k = 0.1
    corners = cv2.goodFeaturesToTrack(cropped, 70, qualityLevel, minDistance, None,
                                      blockSize=blockSize, gradientSize=gradientSize,
                                      useHarrisDetector=useHarrisDetector, k=k)

    winSize = (9, 9)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    corners = cv2.cornerSubPix(cropped, corners, winSize, zeroZone, criteria)
    corners = np.int0(corners)

    return corners


os.chdir("./project_data/G3DCV2020_data_part1_calibration/calib/")
for file in glob.glob("*.png"):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    M, cropped = cropRect(gray)
    invM = cv2.invert(M)[1]
    # croppedColor = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # cropped = cv2.medianBlur(cropped, 9)
    # kernel = np.ones((3, 3), np.uint8)
    # cropped = cv2.dilate(cropped, kernel, iterations=3)

    corners = findChessboardCorners(cropped)
    for i in corners:
        x, y = i.ravel()
        px, py = cv2.perspectiveTransform(np.float32([[[x, y]]]), invM)[0][0]
        cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)

    cv2.imshow(file, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
