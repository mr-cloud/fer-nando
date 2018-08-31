import cv2

img1 = cv2.imread('expressions/models/happy.jpg')
cv2.imshow('img1', img1)
cv2.waitKey(0)
img2 = cv2.imread('expressions/models/happy1.jpg')
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
