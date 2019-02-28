#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import collections as mc

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
            # print(x, y)
            lx = x
            ly = y
            # print(refPt)
            if(len(refPt) == 8):
                refPt = reorder_pairs(refPt)
                print("refPt:", refPt)


# construct the Matrix A
def get_Ai(u,v, u1, v1):
    return [[-1*u, -1*v, -1, 0, 0, 0, u1*u, u1*v, u1],
            [0, 0, 0, -1*u, -1*v, -1, v1*u, v1*v, v1],]


def reorder_pairs(paris):
    p = np.reshape(paris, [4, 2]).copy()
    l2 = []
    result = []
    for i in range(0,4):
        n = np.dot(p[i], p[i])
        print(n)
        l2.append(int(n))

    # print("L2", l2)
    min1 = l2.index(min(l2))
    max1 = l2.index(max(l2))
    result.append(list(p[min1]))

    # print(str(max1))
    t = [0,1,2,3]
    t.remove(min1)
    t.remove(max1)
    x = t[0]
    y = t[1]

    if(p[x][0] < p[y][0]):
        result.append(list(p[x]))
        result.append(list(p[max1]))
        result.append(list(p[y]))
    else:
        result.append(list(p[y]))
        result.append(list(p[max1]))
        result.append(list(p[x]))

    tmp = [item for sublist in result for item in sublist]

    return tmp


def concate_A(u, v, u1, v1):
    final = []
    for i in range(4):
        a = get_Ai(u[i], v[i], u1[i], v1[i])
        final.append(a[0])
        final.append(a[1])
    # print("final:", final)
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
    # print(final_A)
    u, s, vh = np.linalg.svd(final_A)
    v = np.array([vh[8][0:3], vh[8][3:6], vh[8][6:]])
    v = v
    h_inverse = np.linalg.inv(v)

    # print(h_inverse)

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


    cv2.imwrite("out.png", image2)
    cv2.imshow("Transformed image", cv2.imread('out.png', 1))
    cv2.waitKey(0)
    cv2.destroyWindow('Finish Problem3')
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

    # For Problem 4:
    print('problem 4')
    House_3d = np.asarray(
        [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [0, 0, 2], [4, 0, 2], [4, 4, 2], [0, 4, 2], [2, 1, 3], [2, 3, 3]])
    """fig = plt.figure()
    ax = fig.add_subplot(121)
    X = House_3d.T[0]
    Y = House_3d.T[1]
    Z = House_3d.T[2]

    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    """
    M_ext = np.asarray([[-0.707, 0.707, 0, 3], [0.707, -0.707, 0, 0.5], [0, 0, 1, 3]])
    M_int = np.asarray([[100, 0, 200], [-0, 100, 200], [0, 0, 1]])
    M = np.matmul(M_int, M_ext)

    Pt_homo = []
    Pt_2d = []
    for Pt_3d in House_3d:
        Pt_3d = np.concatenate((Pt_3d, [1]), axis=0)
        # For some reason(matrix class), Pt_homo[0]won't return a 1d array
        Pt_homo = np.matmul(M, Pt_3d)
        Pt_2d.append((float(Pt_homo[0]) / Pt_homo[2], float(Pt_homo[1]) / Pt_homo[2]))
    Pt_count = 10
    L_Clct = []
    # Traverse and connect each pairs of points in Pt_2d
    for i in range(0, Pt_count - 2):
        for j in range(i + 1, Pt_count - 1):
            # If two checked points are not same, connect them
            if (Pt_2d[i] != Pt_2d[j]):
                L_Clct.append([Pt_2d[i], Pt_2d[j]])

    # print(L_Clct)
    lc = mc.LineCollection(L_Clct)
    fig, ax = plt.subplots()
    ax.add_collection(lc)

    ax.autoscale()
    plt.show()

