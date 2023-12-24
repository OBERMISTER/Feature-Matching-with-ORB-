# Feature-Matching-with-ORB-
Feature Matching with ORB:
import cv2
import numpy as np

def feature_matching_orb(image1_path, image2_path):
    image1 = cv2.imread(image1_path, 0)
    image2 = cv2.imread(image2_path, 0)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('ORB Feature Matching', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image1_path = 'path/to/your/image1.jpg'
image2_path = 'path/to/your/image2.jpg'
feature_matching_orb(image1_path, image2_path)
