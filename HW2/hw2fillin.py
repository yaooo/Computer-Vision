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
        if(len(refPt) < 8):
            refPt.append(x)
            refPt.append(y)
            print(x, y)
            lx = x
            ly = y


# construct the Matrix A
def get_Ai(u,v, u1, v1):
    return [[-1*u, -1*v, -1, 0, 0, 0, u1*u, u1*v, u1],
            [0, 0, 0, -1*u, -1*v, -1, v1*u, v1*v, v1],]


def reorder_pairs(paris):
    l2 = []
    result = paris.copy()
    for i in paris:
        l2 = [l2, i[0]^2+i[1]^2]
    min = l2.index(min(l2))
    max = l2.index(max(l2))
    result[0] = paris[min]
    result[1] = paris[max]
    paris.pop([min,max])
    paris[0][0]


def concate_A(u, v, u1, v1):
    final = []
    for i in range(4):
        a = get_Ai(u[i], v[i], u1[i], v1[i])
        final.append(a[0])
        final.append(a[1])
    print("final:", final)
    return final



def load_img(name):
    # Read Image
    image = cv2.imread(name, 1)
    # image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, click_and_keep)



    while 1:
        if len(refPt)>8:
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



def display_new_img(name):
    u = refPt[:, 0]
    v = refPt[:, 1]

    u1 = [0, 0, 299, 299]
    v1 = [0, 199, 199, 0]
    final_A = concate_A(u, v, u1, v1)
    final_A = np.vstack(final_A)
    print(final_A)
    u, s, vh = np.linalg.svd(final_A)
    v = np.array([vh[8][0:3], vh[8][3:6], vh[8][6:]])
    v = v
    h_inverse = np.linalg.inv(v)

    print(h_inverse)

    width, height = (300, 200)

    image2 = np.empty((200, 300, 3))
    image = cv2.imread(name)
    for y in range(height):
        for x in range(width):
            x1y1z= np.array([x,y,1]).reshape([3,1])
            t = np.matmul(h_inverse, x1y1z)
            t = t/t[2]

            # print(orignal_cord)
            # print(data[(x,y)])
            y1 = int(t[1])
            x1 = int(t[0])
            # print(int(t[0]), int(t[1]))
            image2[y][x] = image[y1,x1]

    #
    cv2.imshow("Transformed image", image2)
    cv2.imwrite("out.png", image2)
    # image3 = cv2.imread('foo.png', 1)


    # cv2.imshow('New image', cv2.convertScaleAbs(image2))
    # cv2.wait(0)
    # cv2.destroyAllWindows()


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    name = "ts.jpg"
    load_img(name)
    if(len(refPt) >7):
        refPt = np.reshape(refPt, (4, 2))

    display_new_img(name)
# Select End PoinTts of foreshortened window or billboard

# Set the corresponding point in the frontal view as

# Estimate the homography

# Warp the image
# cv.Remap(src, dst, mapx, mapy, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0))

# Crop the image

