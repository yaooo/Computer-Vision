import numpy as np
import cv2
import matplotlib.pyplot as plt
# May need to do: sudo dnf install python3-tkinter (CentOS system)
import sys

def main():
    print('Hello there', sys.argv[1])
    img = cv2.imread('testing.jpg', 1)
    plt.imshow(img, interpolation = 'bicubic')
    # plt.xticks([])
    # plt.yticks([])
    plt.show()
    cv2.destryAllWindows()

if __name__ == '__main__':
    main()