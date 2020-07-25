import cv2
import numpy as np
import open3d as o3d

from utils import (
    loadIntrinsics,
    sortCorners,
    createRays,
    linePlaneIntersection,
    fitPlane,
    outerContour,
    findPlaneFromHomography,
    findPointsInsidePoly
)

rectWidth = 25
rectHeight = 15

kernel4 = np.ones((4, 4), np.uint8)


def findRectanglePatterns(firstFrame):
    """
    Given the first frame, finds the rectangular patterns in the image with float precision
    """
    gray = cv2.cvtColor(firstFrame, cv2.COLOR_RGBA2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    winSize = (16, 16)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 200, 0.1)

    minContourLength = 30
    polys = []
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) >= minContourLength:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            curve = cv2.approxPolyDP(contour, epsilon, True)
            if len(curve) == 4 and cv2.isContourConvex(curve):
                curve = cv2.cornerSubPix(gray, np.float32(curve), winSize, zeroZone, criteria)
                sortedCurve = sortCorners(curve)
                score = outerContour(sortedCurve.astype(np.int32), gray)
                polys.append((sortedCurve, score))

    polys.sort(key=lambda x: x[1], reverse=False)
    return [p[0] for p in polys]


def findReference3DPoints(img, rect, plane, K_inv):
    imgPoints = findPointsInsidePoly(img, rect.astype(np.int32))
    if imgPoints is None:
        return None, None
    homoImgPoints = np.hstack((imgPoints[:, 0], np.ones(imgPoints.shape[0]).reshape(-1, 1),))
    rays = createRays(homoImgPoints, K_inv)
    points3D = [linePlaneIntersection(plane, ray) for ray in rays]
    return points3D, imgPoints


def processFrame(firstFrame, undistorted, K_inv, upperRect, lowerRect, upperPlane, lowerPlane):
    # Prepare the image for processing (thresholding) and find points
    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
    inRange = cv2.inRange(hsv, (150, 20, 80), (200, 255, 255))
    opening = cv2.morphologyEx(inRange, cv2.MORPH_OPEN, kernel4)
    laserPts = cv2.findNonZero(opening)

    # Find reference points on desk and wall
    upper3DPoints, upperImgPoints = findReference3DPoints(opening, upperRect, upperPlane, K_inv)
    lower3DPoints, lowerImgPoints = findReference3DPoints(opening, lowerRect, lowerPlane, K_inv)

    # DEBUG INFO
    if laserPts is not None:
        for p in laserPts:
            cv2.circle(undistorted, (p[0][0], p[0][1]), 1, (0, 0, 255))
    cv2.imshow('undistorted', undistorted)

    # Then fit a plane if we have enough points
    if upper3DPoints is not None and lower3DPoints is not None:
        referencePoints = np.array(upper3DPoints + lower3DPoints)
        laserPlane = fitPlane(referencePoints)

        homoImgPoints = np.hstack((laserPts[:, 0], np.ones(laserPts.shape[0]).reshape(-1, 1),))
        rays = createRays(homoImgPoints, K_inv)
        points3D = [linePlaneIntersection(laserPlane, ray) for ray in rays]
        x = laserPts.squeeze(1)
        colors = np.flip(firstFrame[x[:, 1], x[:, 0]].astype(np.float64) / 255.0, axis=1)
        return points3D, colors, laserPlane
    return None, None, None


def plotPlane(plt3d, plane, color):
    origin, normal = plane
    xx, yy = np.meshgrid(range(-2000, 2000, 1000), range(-2000, 2000, 1000))
    d = -origin.dot(normal)
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    plt3d.plot_surface(xx, yy, z, alpha=0.2, color=color)


def run(debug=True):
    K, dist = loadIntrinsics()
    K_inv = np.linalg.inv(K)

    cap = cv2.VideoCapture("./project_data/G3DCV2020_data_part2_video/cup1.mp4")
    # cap = cv2.VideoCapture("./project_data/G3DCV2020_data_part3_video/puppet.mp4")
    # cap = cv2.VideoCapture("./project_data/G3DCV2020_data_part4_video/soap.mp4")
    firstFrameDistorted = cap.read()[1]
    firstFrame = cv2.undistort(firstFrameDistorted, K, dist)

    polys = findRectanglePatterns(firstFrame)
    upperRect, lowerRect = polys[0:2]

    # if debug:
    #     cv2.drawContours(firstFrame, [upperRect.astype(np.int32), lowerRect.astype(np.int32)], -1, (0, 0, 255))
    #     cv2.imshow("debug rect contours", firstFrame)
    #     cv2.waitKey(1)

    upperDestPoints: np.ndarray = np.array([[[0, rectHeight]], [[0, 0]], [[rectWidth, 0]], [[rectWidth, rectHeight]]])
    upperRectHomo = cv2.findHomography(sortCorners(upperDestPoints), upperRect)[0]
    upperPlane = findPlaneFromHomography(upperRectHomo, K_inv)

    lowerDestPoints: np.ndarray = np.array([[[0, rectHeight]], [[0, 0]], [[rectWidth, 0]], [[rectWidth, rectHeight]]])
    lowerRectHomo = cv2.findHomography(sortCorners(lowerDestPoints), lowerRect)[0]
    lowerPlane = findPlaneFromHomography(lowerRectHomo, K_inv)

    objPoints = []
    objColors = []

    isRecording = True
    while cap.isOpened():
        if isRecording:
            ret, frame = cap.read()
            if not ret:
                break

        undistorted = cv2.undistort(frame, K, dist)
        framePts, frameColors, laserPlane = processFrame(firstFrame, undistorted, K_inv, upperRect, lowerRect,
                                                         upperPlane,
                                                         lowerPlane)
        if framePts is not None and frameColors is not None:
            objPoints.extend(framePts)
            objColors.extend(frameColors)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
            isRecording = False
        if cv2.waitKey(1) & 0xFF == ord('c'):  # Continue
            isRecording = True

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(objPoints).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(objColors))
    o3d.io.write_point_cloud("output.ply", pcd)
    o3d.visualization.draw_geometries([pcd])

    cap.release()
    cv2.destroyAllWindows()


run()
