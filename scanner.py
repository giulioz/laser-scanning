import cv2
import numpy as np
import open3d as o3d
import sys
import getopt

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

# The reference fiducial rectangles sizes
rectWidth = 25
rectHeight = 15

# HSV ranges for laser line detection
hsvMin = (150, 20, 78)
hsvMax = (200, 255, 255)

# Kernels for opening-closing
kernel2 = np.ones((2, 2), np.uint8)
kernel4 = np.ones((4, 4), np.uint8)


def findRectanglePatterns(firstFrame):
    """
    Given the first frame, finds the 2D rectangular patterns in the image with float precision.
    """

    # Threshold the image using Otsu algorithm
    gray = cv2.cvtColor(firstFrame, cv2.COLOR_RGBA2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find all the possible contours in thresholded image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    winSize = (16, 16)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 200, 0.1)
    minContourLength = 30
    polys = []
    for contour in contours:
        if len(contour) >= minContourLength:
            # We approximate a polygon, we are only interested in rectangles (4 points, convex)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            curve = cv2.approxPolyDP(contour, epsilon, True)
            if len(curve) == 4 and cv2.isContourConvex(curve):
                # We use cornerSubPix for floating point refinement
                curve = cv2.cornerSubPix(gray, np.float32(
                    curve), winSize, zeroZone, criteria)
                sortedCurve = sortCorners(curve)
                score = outerContour(sortedCurve.astype(np.int32), gray)
                polys.append((sortedCurve, score))

    # We sort the found rectangles by score descending, using outerContour function
    # The function calculates the mean of the border inside the rectangle: it must be around full black
    # It's safe to take the first entry as a valid pattern if it exists
    polys.sort(key=lambda x: x[1], reverse=False)
    return [p[0] for p in polys]


def findReference3DPoints(img, rect, plane, K_inv):
    """
    Given the thresholded laser, a rectangle for bounds, the rectangle plane and the inverse
    camera matrix, finds all the 3D points inside the rectangle intersecting the plane.
    This is used to find the reference 3D points inside the wall and desk rectangles.

              laser
     wall   /
       |  /
       x
    /  |_____ desk
    """

    # Find the 2D points inside the rectangle
    imgPoints = findPointsInsidePoly(img, rect.astype(np.int32))
    if imgPoints is None:
        return None, None

    # Create rays and find the intersections with plane
    homoImgPoints = np.hstack(
        (imgPoints[:, 0], np.ones(imgPoints.shape[0]).reshape(-1, 1),))
    rays = createRays(homoImgPoints, K_inv)
    points3D = [linePlaneIntersection(plane, ray) for ray in rays]
    return points3D, imgPoints


def processFrame(firstFrame, undistorted, K_inv, upperRect, lowerRect, upperPlane, lowerPlane, debug=False):
    # Prepare the image for processing (thresholding) and find points
    hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
    inRange = cv2.inRange(hsv, hsvMin, hsvMax)
    final = cv2.morphologyEx(inRange, cv2.MORPH_OPEN, kernel4)
    laserPts = cv2.findNonZero(final)

    # DEBUG INFO
    if debug:
        if laserPts is not None:
            for p in laserPts:
                cv2.circle(undistorted, (p[0][0], p[0][1]), 1, (0, 0, 255))
        cv2.imshow('undistorted', undistorted)

    # Find reference points on desk and wall
    upper3DPoints, upperImgPoints = findReference3DPoints(
        final, upperRect, upperPlane, K_inv)
    lower3DPoints, lowerImgPoints = findReference3DPoints(
        final, lowerRect, lowerPlane, K_inv)

    # Then fit a plane if we have enough points
    if upper3DPoints is not None and lower3DPoints is not None:
        # Find the corrisponding laser plane
        referencePoints = np.array(upper3DPoints + lower3DPoints)
        laserPlane = fitPlane(referencePoints)

        # Find 3D points with line-plane intersection
        homoImgPoints = np.hstack(
            (laserPts[:, 0], np.ones(laserPts.shape[0]).reshape(-1, 1),))
        rays = createRays(homoImgPoints, K_inv)
        points3D = [linePlaneIntersection(laserPlane, ray) for ray in rays]

        # Recover colors for points from first frame
        x = laserPts.squeeze(1)
        colors = np.flip(firstFrame[x[:, 1], x[:, 0]].astype(
            np.float64) / 255.0, axis=1)
        return points3D, colors, laserPlane
    return None, None, None


def run(path, debug=False):
    # Load our camera parameters
    K, dist = loadIntrinsics()
    K_inv = np.linalg.inv(K)

    # Load our video and read the first frame
    # We will use the first frame to find the reference rectangles and colors for the point cloud
    cap = cv2.VideoCapture(path)
    firstFrameDistorted = cap.read()[1]
    firstFrame = cv2.undistort(firstFrameDistorted, K, dist)

    # Finds the reference rectangles in the first frame
    polys = findRectanglePatterns(firstFrame)
    upperRect, lowerRect = polys[0:2]

    # DEBUG INFO
    if debug:
        firstFrameDbg = firstFrame.copy()
        cv2.drawContours(firstFrameDbg, [upperRect.astype(
            np.int32), lowerRect.astype(np.int32)], -1, (0, 0, 255))
        cv2.imshow("debug rect contours", firstFrameDbg)
        cv2.waitKey(1)

    # Find the WALL rectangle homography and plane
    upperDestPoints: np.ndarray = np.array(
        [[[0, rectHeight]], [[0, 0]], [[rectWidth, 0]], [[rectWidth, rectHeight]]])
    upperRectHomo = cv2.findHomography(
        sortCorners(upperDestPoints), upperRect)[0]
    upperPlane = findPlaneFromHomography(upperRectHomo, K_inv)

    # Find the DESK rectangle homography and plane
    lowerDestPoints: np.ndarray = np.array(
        [[[0, rectHeight]], [[0, 0]], [[rectWidth, 0]], [[rectWidth, rectHeight]]])
    lowerRectHomo = cv2.findHomography(
        sortCorners(lowerDestPoints), lowerRect)[0]
    lowerPlane = findPlaneFromHomography(lowerRectHomo, K_inv)

    # Here we store our found 3D cloud
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
                                                         lowerPlane, debug)
        if framePts is not None and frameColors is not None:
            objPoints.extend(framePts)
            objColors.extend(frameColors)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):  # Pause
            isRecording = False
        if cv2.waitKey(1) & 0xFF == ord('c'):  # Continue
            isRecording = True

    # Finally, save our point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.vstack(objPoints).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(objColors))
    o3d.io.write_point_cloud("output.ply", pcd)
    o3d.visualization.draw_geometries([pcd])

    cap.release()
    cv2.destroyAllWindows()


opts, args = getopt.getopt(sys.argv, "v")
debug = args.count("-v") > 0
path = args[-1]
run(path, debug)
