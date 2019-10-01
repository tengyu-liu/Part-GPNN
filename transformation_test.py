import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

# create image and bboxes
h = 600
w = 900
img = np.ones([h,w,3])
bboxes = []
for i in range(5):
    x0 = random.randint(0, h)
    x1 = random.randint(min(x0 + 50, h), min(x0 + 100, h))
    y0 = random.randint(0, w)
    y1 = random.randint(min(y0 + 50, w), min(y0 + 100, w))
    img[x0:x1,y0:y1,:] = 0
    bboxes.append([x0,x1,y0,y1])

# apply transformation
if random.random() < 0.2:
    # shear
    pass
if random.random() < 0.2:
    # rotate
    pass
if random.random() < 0.2:
    # scale
    pass
if random.random() < 0.2:
    # horizontal flip
    pass
if random.random() < 0.2:
    # hue
    pass
if random.random() < 0.2:
    # brightness
    pass


def draw_img(img, bboxes):
    plt.imshow(img)
    for x0,x1,y0,y1 in bboxes:
        plt.plot([y0,y1,y1,y0,y0], [x0,x0,x1,x1,x0], c='red')
    plt.show()

draw_img(img, bboxes)
