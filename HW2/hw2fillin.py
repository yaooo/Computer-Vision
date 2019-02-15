#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt

# import modules used here -- sys is a very standard one
import sys

refPt = []
image = np.zeros((512, 512, 3), np.uint8)
windowName = 'HW Window'
lx = -1
ly = -1


def click_and_keep(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, image, lx, ly

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        if(len(refPt) < 4):
            refPt.append((x, y))
            print(x, y)
            lx = x
            ly = y


# Gather our code in a main() function
def main():
    # Read Image
    image = cv2.imread('ts.jpg', 1)
    # image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, click_and_keep)


    while 1:
        if len(refPt)>4:
            color = (0, 255, 255)
        else:
            color = (255, 0, 255)

        image = cv2.circle(image, (lx, ly), 10, color, -1);
        cv2.imshow(windowName, image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break

    # Close the window will exit the program
    cv2.destroyAllWindows()


def display_new_img():
    length = 300
    width = 200
    imshape = (width, length, 3)

    image2 = np.empty(imshape)

    cv2.imshow('New image', cv2.convertScaleAbs(image2))
    cv2.waitKey(0)
    cv2.destroyWindow('Problem2_transferredImage')


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
    display_new_img()
# Select End Points of foreshortened window or billboard

# Set the corresponding point in the frontal view as

# Estimate the homography

# Warp the image
# cv.Remap(src, dst, mapx, mapy, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))

# Crop the image

