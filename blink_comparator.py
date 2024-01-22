#operating system module: used to list the content of a folder
import os

#pathlib module: used to simplify working with files and folders
#   this is because the os module treats paths as strings and this can be difficult to handle
#   instead the pathlib module treats paths as objects
from pathlib import Path

import numpy as np
import math

#to parse the input arguments
import argparse

#OpenCV module: used for working with images
import cv2 as cv

#A keypoint is a feature used to characterize an image, in our case these features are stars
MIN_NUM_KEYPOINT_MATCHES=50


#Import function to count elements in the sky
from count_elements import count_elements_in_the_sky



#Function that finds the best keypoint matches between each pair of images
def find_best_matches(img1,img2):
    
    #Create ORB object
    #   ORB is an object that embeddes algorithms for keypoints finding
    #   The variable nfeatures specifies the number of keypoints (expressed as features in a matrix)
    orb = cv.ORB_create(nfeatures=100)
    
    #We calculate the keypoints in each image
    kp1, desc1 = orb.detectAndCompute(img1, mask=None)
    kp2, desc2 = orb.detectAndCompute(img2, mask=None)

    #Create BFMatcher object
    #   BFMatcher is an object containing algorithms for finding keypoints common to both images
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    #Here we calculate the keypoints common to both images
    #   The similarity between keypoints is evaluated using the Hamming distance
    matches = bf.match(desc1,desc2)

    #Here we find the best matching keypoints by sorting them accordingly to the previously calculated distance
    matches = sorted(matches, key=lambda x: x.distance)

    #From the list of best keypoints we take only the fraction defined in MIN_NUM_KEYPOINT_MATCHES
    best_matches = matches[:MIN_NUM_KEYPOINT_MATCHES]

    return kp1, kp2, best_matches

def QC_best_matches(img_match):
    """Draw img_match to the screen"""
    #img_match consists of the left and right images with keypoits drawn as circles and with colered lines connecting the corresponding keypoints

    cv.imshow("Best {} Matches".format(MIN_NUM_KEYPOINT_MATCHES), img_match)
    #Keep the window open for 2.5 seconds
    cv.waitKey(2500)
    #cv.waitKey does not close the window

def register_image(img1, img2, kp1, kp2, best_matches):
    """Return first image registered to second image"""

    if(len(best_matches) >= MIN_NUM_KEYPOINT_MATCHES):

        #Initialize two arrays with as many rows as there are best matches
        # the arrays will be at least 50x2
        src_pts = np.zeros((len(best_matches), 2), dtype=np.float32)
        dst_pts = np.zeros((len(best_matches), 2), dtype=np.float32)
        
        for i, match in enumerate(best_matches):
            src_pts[i, :] = kp1[match.queryIdx].pt
            dst_pts[i, :] = kp2[match.trainIdx].pt

        #Homography is a transformation that maps points in one image to corresponding points in another image
        # RANSAC is an outlier detector to increase accurancy
        h_array, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

        
        height, width = img2.shape

        #Warp the first image so that it perfectly aligns with the second image
        img1_warped = cv.warpPerspective(img1, h_array, (width, height))

        return img1_warped

    else:

        #This error happens, for example, when the two images are not well alligned
        print("WARNING: Number of keypoints matches is less than {}\n".format(MIN_NUM_KEYPOINT_MATCHES))
        return img1

def blink(image_1, image_2, window_name, num_loops):
    """Rapidly alternate images simulating a blink comparator"""
    for _ in range(num_loops):
        cv.imshow(window_name, image_1)
        cv.waitKey(330)
        cv.imshow(window_name, image_2)
        cv.waitKey(330)

def main():
    image_path1="1_bright_transient_left.png"
    image_path2="1_bright_transient_right.png"

        #Load an image from night 1 (left) and from night 2 (right), as greyscale images
        #   paths are converted to string for the imread() method
        #   images are also converted to grayscale in order to work with only one channel (intensity)
        #img1 = cv.imread(str(path1 / night1_files[i]), cv.IMREAD_GRAYSCALE)
        #img2 = cv.imread(str(path2 / night2_files[i]), cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(image_path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)
    #To keep track on what is going on we print a message indicating which files are being compared
    print("Comparing {} to {} \n".format(image_path1,image_path2))

    number_of_celestial_bodies = math.ceil((count_elements_in_the_sky(image_path1) + count_elements_in_the_sky(image_path2))/2)
        
    print("Approximate number of celestial bodies: {}\n".format(number_of_celestial_bodies))      

        #The find_best_matches() function returns:
        # kp1: keypoints for image on the left
        # kp2: keypoints for image on the right
        # best_matches: list of matching keypoints
    kp1, kp2, best_matches = find_best_matches(img1,img2)


        #We keep track on the matches by drawing them
        #   img_match consists of the left and right images with keypoits drawn as circles and with colered lines connecting the corresponding keypoints
        #   the output argument is set to None since we only look at the output image, not save it to a file  
    img_match = cv.drawMatches(img1, kp1, img2, kp2, best_matches, outImg=None)

        #To distinguish between the two images we draw a vertical white line on the right of img1
        #   First we get the dimension of img1
    height, width = img1.shape
        #   We pass to line() the image on which we want to draw (img_match), the start condition, the end condition, the line color, the thickness
    cv.line(img_match, (width, 0), (width, height), (255, 255, 255), 1)

        #This function display img_match
    QC_best_matches(img_match)

        #With the best keypoints matches found and checked, it is time to register the first image to the second
    img1_registered = register_image(img1, img2,kp1, kp2, best_matches)


    blink(img1_registered, img2, "Blink Comparator", num_loops=15)

    cv.destroyAllWindows()

if( __name__ == '__main__'):

    parser = argparse.ArgumentParser(description='A Blink Comparator is an optical device used to rapidly alternate between two images of the same area of the sky, aiding in the detection of differences such as the presence of celestial objects that have changed or moved')
    parser.add_argument('LeftImage', type=str, help='String path of the left image from the first night')
    parser.add_argument('RightImage', type=str, help='String path of the right image from the second night')

    args = parser.parse_args()


    main()
