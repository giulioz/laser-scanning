import cv2
import numpy as np
import math


def outerContour(contour, gray, margin=10):
    """
    Given a contour and an image, returns the mean of the pixels around the contour.
    This is used to detect the rectangle fiducial pattern.
    """
    # We create two masks, one with the poly and one with the poly eroded
    kernel = np.ones((margin, margin), np.uint8)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, contour, 255)
    eroded = cv2.erode(mask, kernel)
    mask = cv2.bitwise_xor(eroded, mask)

    # We calculate the mean with the two XORed mask
    mean = cv2.mean(gray, mask)
    return mean[0]


def loadIntrinsics(path="intrinsics.xml"):
    """
    Loads camera intrinsics from an xml file. Uses a default path if not provided (intrinsics.xml).
    """
    intrinsics = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = intrinsics.getNode("K").mat()
    dist = intrinsics.getNode("dist").mat()
    return K, dist


def sortCorners(corners):
    """
    Sorts an array of corners clockwise.
    """
    center = np.sum(corners, axis=0) / len(corners)

    # Returns the point rotation angle in radians from the center
    def rot(point):
        return math.atan2(point[0][0] - center[0][0], point[0][1] - center[0][1])

    sortedCorners = sorted(corners, key=rot, reverse=True)
    return np.roll(sortedCorners, 2, axis=0)


def findPointsInsidePoly(img, poly):
    """
    Get the positions of all points inside a given poly from a binary image.
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    imgMasked = np.bitwise_and(img, mask)
    points = cv2.findNonZero(imgMasked)
    return points


def createRays(pts, K_inv):
    """
    Transforms an array of points into an array of 3D rays, using the inverse intrinsics K_inv.
    """
    # We simply need to multiply per K_inv, this way we get a 3D direction.
    # We can get the points of the ray multiplying the direction with a scalar.
    return [np.matmul(K_inv, p) for p in pts]


def linePlaneIntersection(plane, rayDir):
    """
    Calculate the 3D intersection between a plane and a ray, returning a 3D point.
    """
    pOrigin, pNormal = plane
    d = np.dot(pOrigin, pNormal) / np.dot(rayDir, pNormal)
    return rayDir * d


def findPlaneFromHomography(H, K_inv):
    """
    Some black magic to find a plane (origin and normal) from an homography and an inverse intrinsics matrix.
    """
    # First, we apply the inverse intrinsics to the homography to remove the camera effects
    result = np.matmul(K_inv, H)

    # We need to normalize our matrix to remove the scale factor
    result /= cv2.norm(result[:, 1])

    # We split our resulting homography columns to get the two 2D rotation basis vectors and translation
    r0, r1, t = np.hsplit(result, 3)

    # To get the third rotation basis vector we simply make the cross product between the 2D basis
    r2 = np.cross(r0.T, r1.T).T

    # Since r0 and r1 may not be orthogonal, we use the Zhang aproximation:
    # we keep only the u and vt part of the SVD of the new rotation basis matrix
    # to minimize the Frobenius norm of the difference
    _, u, vt = cv2.SVDecomp(np.hstack([r0, r1, r2]))
    R = np.matmul(u, vt)

    # We finally have our origin center and normal vector
    origin = t[:, 0]
    normal = R[:, 2]
    return origin, normal


def fitPlane(points):
    """
    Fit a plane from a set of 3D points, as described in "Least Squares Fitting of Data by Linear or Quadratic Structures".
    """
    centroid = np.mean(points, axis=0)
    xxSum = 0
    xySum = 0
    xzSum = 0
    yySum = 0
    yzSum = 0
    zzSum = 0

    for point in points:
        diff = point - centroid
        xxSum += diff[0] * diff[0]
        xySum += diff[0] * diff[1]
        xzSum += diff[0] * diff[2]
        yySum += diff[1] * diff[1]
        yzSum += diff[1] * diff[2]
        zzSum += diff[2] * diff[2]

    detX = yySum * zzSum - yzSum * yzSum
    detY = xxSum * zzSum - xzSum * xzSum
    detZ = xxSum * yySum - xySum * xySum
    detMax = max(detX, detY, detZ)

    if detMax == detX:
        normal = np.array([detX, xzSum * yzSum - xySum * zzSum, xySum * yzSum - xzSum * yySum])
    elif detMax == detY:
        normal = np.array([xzSum * yzSum - xySum * zzSum, detY, xySum * xzSum - yzSum * xxSum])
    else:
        normal = np.array([xySum * yzSum - xzSum * yySum, xySum * xzSum - yzSum * xxSum, detZ])

    normal = normal / np.linalg.norm(normal)
    origin = np.array(centroid)
    return origin, normal
