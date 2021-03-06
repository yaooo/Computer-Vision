import numpy as np
import cv2
import matplotlib.pyplot as plt
from sympy import *
# May need to do: sudo dnf install python3-tkinter (CentOS system)
import sys

def Q1():
    print("Question 1:")
    print('Hello there')
    img = cv2.imread('testing.jpg', 1)
    plt.imshow(img, interpolation = 'bicubic')
    # plt.xticks([])
    # plt.yticks([])
    plt.show()

def Q2():
    print("\nQuesetion 2:")
    print("Yes, because the vectors are independent of each other")

def Q3():
    print("\nQuesetion 3:")
    print("Yes, because t1 and t2 are not linearly independent")

def Q4():
    print("\nQuesetion 4:")
    print("Yes, because w1, w2, and w3 are independent and basis of R3")


def Q5():
    print("\nQuestion 5:")
    print("Displaying the plot...")
    x = [0, 1, 1, 3, 3, 5]
    y = [1, 3.2, 5, 7.2, 9.3, 11.1]
    z = np.polyfit(x, y, 1)

    # use poly1d dealing with polynomials
    p = np.poly1d(z)

    xp = np.linspace(-1, 6, 100)
    plt.plot(x, y, '.', xp, p(xp), '-',)
    plt.show()


def Q6():
    print("\nQuestion 6:")
    A = [[4.29, 2.2, 5.51], [5.20, 10.1, -8.24], [1.33, 4.8, -6.62]]
    u, s, vh = np.linalg.svd(A)
    print("The conjecture is suppored because the third singular value of A is 0.0164 (about 0)")
    print("0.0164 is close to 0, and it could be resulted from the rounding error.")


if __name__ == '__main__':
    Q1()
    Q2()
    Q3()
    Q4()
    Q5()
    Q6()