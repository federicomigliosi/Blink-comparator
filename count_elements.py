import numpy as np
import cv2 as cv


def count_elements_in_the_sky(image_path):
    #image_path is a string rapresenting the path to the input image
    image=cv.imread(image_path)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #invert image
    gray_image = np.invert(gray_image)

    blur = cv.GaussianBlur(gray_image,(5,5),0)
    _,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    return(len(contours)-1)






