import cv2
import numpy as np
import math


def outerContour(contour, gray):
    margin = 10
    kernel = np.ones((margin, margin), np.uint8)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    eroded = cv2.erode(mask, kernel)
    mask = cv2.bitwise_xor(eroded, mask)
    mean = cv2.mean(gray, mask)
    return mean[0]


def loadIntrinsics():
    intrinsics = cv2.FileStorage("intrinsics.xml", cv2.FILE_STORAGE_READ)
    K = intrinsics.getNode("K").mat()
    dist = intrinsics.getNode("dist").mat()
    return K, dist


def sortCorners(corners: np.ndarray):
    center = np.sum(corners, axis=0) / 4
    sortedCorners = sorted(
        corners,
        key=lambda p: math.atan2(p[0][0] - center[0][0], p[0][1] - center[0][1]),
        reverse=True,
    )
    return np.roll(sortedCorners, 2, axis=0)


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


def findBestLine(mask):
    lines = cv2.HoughLines(mask, 1, np.pi / 180, 50)
    if lines is None:
        return None
    else:
        return lines[0][0]


def drawTrackingLine(img, line):
    if line is not None:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
