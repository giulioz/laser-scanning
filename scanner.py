import math
import cv2
import numpy as np
import open3d as o3d

rectWidth = 23*20
rectHeight = 13*20

kernel4 = np.ones((4, 4), np.uint8)


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


def sortCorners(corners: np.ndarray):
    center = np.sum(corners, axis=0) / 4
    sorted_corners = sorted(
        corners,
        key=lambda p: math.atan2(p[0][0] - center[0][0], p[0][1] - center[0][1]),
        reverse=True,
    )
    return np.roll(sorted_corners, 2, axis=0)


def findRectanglesHomographies(firstFrame, K, dist):
    firstFrame = cv2.undistort(firstFrame, K, dist)
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
                polys.append(sortCorners(curve.astype(np.int)))

    polys.sort(key=lambda x: outerContour(x, gray), reverse=False)
    upperRect, lowerRect = polys[0:2]

    upperDestPoints: np.ndarray = np.array([[[0, rectHeight]], [[0, 0]], [[rectWidth, 0]], [[rectWidth, rectHeight]]])
    upperRectHomo = cv2.findHomography(upperRect, sortCorners(upperDestPoints))[0]

    lowerDestPoints: np.ndarray = np.array(
        [[[0, rectHeight]], [[0, 0]], [[rectWidth, 0]], [[rectWidth, rectHeight]]])
    lowerRectHomo = cv2.findHomography(lowerRect, sortCorners(lowerDestPoints))[0]
    return upperRect, lowerRect, upperRectHomo, lowerRectHomo


def findPlaneFromHomography(H, K_inv):
    result = np.matmul(K_inv, H)
    result /= cv2.norm(result[:, 1])
    r0, r1, t = np.hsplit(result, 3)
    r2 = np.cross(r0.T, r1.T).T
    _, u, vt = cv2.SVDecomp(np.hstack([r0, r1, r2]))
    R = np.matmul(u, vt)
    origin = t[:, 0]
    normal = R[:, 2]
    return origin, normal


# def findBestLine(mask):
#     lines = cv2.HoughLines(mask, 1, np.pi / 180, 50)
#     if lines is None:
#         return None
#     else:
#         return lines[0][0]


# def drawTrackingLine(img, line):
#     if line is not None:
#         rho, theta = line
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 2000 * (-b))
#         y1 = int(y0 + 2000 * (a))
#         x2 = int(x0 - 2000 * (-b))
#         y2 = int(y0 - 2000 * (a))
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def findPointsInsidePoly(img, poly):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    imgMasked = np.bitwise_and(img, mask)
    points = cv2.findNonZero(imgMasked)
    return points


def createRays(pts, K_inv):
    return [np.matmul(K_inv, p) for p in pts]


def linePlaneIntersection(plane, rayDir):
    pOrigin, pNormal = plane
    d = np.dot(pOrigin, pNormal) / np.dot(rayDir, pNormal)
    return rayDir * d


def findReference3DPoints(img, rect, plane, K_inv):
    imgPoints = findPointsInsidePoly(img, rect)
    if imgPoints is None:
        return None, None
    homoImgPoints = np.hstack((imgPoints[:, 0], np.ones(imgPoints.shape[0]).reshape(-1, 1),))
    rays = createRays(homoImgPoints, K_inv)
    points3D = [linePlaneIntersection(plane, ray) for ray in rays]
    return points3D, imgPoints


def fitPlane(points):
    mean = np.mean(points, axis=0)
    xx = 0
    xy = 0
    xz = 0
    yy = 0
    yz = 0
    zz = 0

    for point in points:
        diff = point - mean
        xx += diff[0] * diff[0]
        xy += diff[0] * diff[1]
        xz += diff[0] * diff[2]
        yy += diff[1] * diff[1]
        yz += diff[1] * diff[2]
        zz += diff[2] * diff[2]

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy
    det_max = max(det_x, det_y, det_z)

    if det_max == det_x:
        normal = np.array([det_x, xz * yz - xy * zz, xy * yz - xz * yy])
    elif det_max == det_y:
        normal = np.array([xz * yz - xy * zz, det_y, xy * xz - yz * xx])
    else:
        normal = np.array([xy * yz - xz * yy, xy * xz - yz * xx, det_z])

    normal = normal / np.linalg.norm(normal)
    origin = np.array(mean)
    return origin, normal


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
    # if laserPts is not None:
    #     for p in laserPts:
    #         cv2.circle(undistorted, (p[0][0], p[0][1]), 1, (0, 0, 255))
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
        return points3D, colors
    return None, None


def run():
    cap = cv2.VideoCapture("./project_data/G3DCV2020_data_part2_video/cup1.mp4")
    firstFrame = cap.read()[1]

    K, dist = load_intrinsics()
    K_inv = np.linalg.inv(K)

    upperRect, lowerRect, upperRectHomo, lowerRectHomo = findRectanglesHomographies(firstFrame, K, dist)
    upperPlane = findPlaneFromHomography(upperRectHomo, K_inv)
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
        framePts, frameColors = processFrame(firstFrame, undistorted, K_inv, upperRect, lowerRect, upperPlane, lowerPlane)
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
