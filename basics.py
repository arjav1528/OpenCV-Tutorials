from _ast import Lambda
from types import LambdaType

import cv2
import random
import numpy as np

# Tut 1
# img = cv2.imread('assets/logo.jpg', 1)
# img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# cv2.imwrite('assets/new_img.jpg', img)
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Tut 2
# img = cv2.imread('assets/logo.jpg', -1)
# cv2.resize(img, (0,0), fx=2, fy=2)

# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
#
# print(img.shape)
#
# tag = img[100:200, 60:90]
# img[10:110,50:80] = tag
# cv2.imwrite('assets/new.jpg', tag)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Tut 3

# cap = cv2.VideoCapture(0)
#
# while(True):
#     ret, frame = cap.read()
#     width = int(cap.get(3))
#     height = int(cap.get(4))
#
#     image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
#     smaller_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
#
#     image[:height//2, :width//2] = smaller_frame
#     image[height // 2:, :width // 2] = smaller_frame
#     image[:height // 2, width // 2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
#     image[height // 2:, width // 2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
#
#     cv2.imshow('frame', image)
#
#     if(cv2.waitKey(1) == ord('q')):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# Tut 4

# cap = cv2.VideoCapture(0)
#
# while True:
#     ret,frame = cap.read()
#     width = int(cap.get(3))
#     height = int(cap.get(4))
#
#     img = cv2.line(frame,(0,0), (width,height),(255,255,255),2)
#
#     cv2.imshow('frame', img)
#
#     if(cv2.waitKey(1) == ord('q')):
#         break
#
#
#
#
#
# cap.release()
# cv2.destroyAllWindows()



# Tut 5

# cap = cv2.VideoCapture(0)
#
# while True:
#     ret,frame = cap.read()
#     height, width = (cap.get(3), cap.get(4))
#
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     lower_blue = np.array([230, 245, 235])
#     upper_blue = np.array([255, 255, 255])
#
#     mask = cv2.inRange(frame, lower_blue, upper_blue)
#
#     # result = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.imshow('frame',mask)
#
#
#
#     if(cv2.waitKey(1) == ord('q')):
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()


# Tut 6


img = cv2.imread('assets/img.png')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


corners = cv2.goodFeaturesToTrack(grey,100,0.6,0)
corners = np.int_(corners)

# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(img, (int(x), int(y)), 5, (255,200,100), -1)
#
#
#
# cv2.imshow('image', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()



for i in range (len(corners)):
    for j in range (i+1,len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])

        color = (tuple(map(lambda x: int(x), np.random.randint(0, 255, 3))))

        cv2.line(img,corner1,corner2,color,1)


cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


