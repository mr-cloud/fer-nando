# -*- coding: utf-8 -*-

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np


img1 = cv2.imread('expressions/models/h4.jpg')
cv2.imshow('img1', img1)
cv2.waitKey(0)
img2 = cv2.imread('expressions/models/h4.jpg')
fontpath = "fonts/simsun.ttc"  # <== 这里是宋体路径
font = ImageFont.truetype(fontpath, 30)
img_pil = Image.fromarray(img2)
draw = ImageDraw.Draw(img_pil)
height, width, channels = img2.shape
draw.text((width / 2, 20), u'中文', font=font,
          fill=(0, 255, 0, 0))
img2 = np.array(img_pil)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
