import glob, os
import cv2
import numpy as np


def load_intrinsics():
    intrinsics = cv2.FileStorage("intrinsics.xml", cv2.FILE_STORAGE_READ)
    K = intrinsics.getNode("K").mat()
    dist = intrinsics.getNode("dist").mat()
    return K, dist

def outerContour(contour, gray):
    margin = 10
    kernel = np.ones((margin, margin), np.uint8)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    eroded = cv2.erode(mask, kernel)
    mask = cv2.bitwise_xor(eroded, mask)
    mean = cv2.mean(gray, mask)
    return mean[0]


cap = cv2.VideoCapture("./project_data/G3DCV2020_data_part2_video/cup1.mp4")
K, dist = load_intrinsics()


def findRectangles():
    firstFrame = cap.read()[1]
    firstFrame = cv2.undistort(firstFrame, K, dist)
    gray = cv2.cvtColor(firstFrame, cv2.COLOR_RGBA2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    minContourLength = 30
    polys = []
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) >= minContourLength:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            curve = cv2.approxPolyDP(contour, epsilon, True)
            if len(curve) == 4 and cv2.isContourConvex(curve):
                polys.append(curve)

    polys.sort(key=lambda x: outerContour(x, gray), reverse=False)
    rectangles = polys[0:2]

    # cv2.drawContours(firstFrame, contours, -1, (0, 0, 255))
    # cv2.drawContours(firstFrame, rectangles, -1, (255, 0, 0))
    # cv2.imshow('firstFrame', firstFrame)
    # cv2.waitKey(0)

    return rectangles


def processVideo(rectangles):
    kernel = np.ones((4, 4), np.uint8)
    isRecording = True
    while cap.isOpened():
        if isRecording:
            ret, frame = cap.read()
            if not ret:
                break

        frame = cv2.undistort(frame, K, dist)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        inRange = cv2.inRange(hsv, (150, 20, 80), (200, 255, 255))
        opening = cv2.morphologyEx(inRange, cv2.MORPH_OPEN, kernel)

        cv2.drawContours(opening, rectangles, -1, 255)

        cv2.imshow('inRange', opening)
        cv2.imshow('hsv', hsv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
            isRecording = False
        if cv2.waitKey(1) & 0xFF == ord('c'):  # Continue
            isRecording = True


rectangles = findRectangles()
processVideo(rectangles)

cap.release()
cv2.destroyAllWindows()
