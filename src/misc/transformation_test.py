import copy
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, color

def draw_img(img, bboxes):
    plt.imshow(img)
    for x0,x1,y0,y1 in bboxes:
        plt.scatter([y0,y1,y1,y0,y0], [x0,x0,x1,x1,x0], c='red')

def draw_img2(img, bboxes):
    plt.imshow(img)
    for p1,p2,p3,p4 in bboxes:
        plt.plot([p1[1],p2[1],p3[1],p4[1],p1[1]], [p1[0],p2[0],p3[0],p4[0],p1[0]], c='red')

# create image and bboxes
h = 600
w = 900
img = np.ones([h,w,3])
img[:,:,0] = random.random()
img[:,:,1] = random.random()
img[:,:,2] = random.random()

bboxes = []
for i in range(5):
    x0 = random.randint(0, h)
    x1 = random.randint(min(x0 + 50, h), min(x0 + 100, h))
    y0 = random.randint(0, w)
    y1 = random.randint(min(y0 + 50, w), min(y0 + 100, w))
    img[x0:x1,y0:y1,:] = np.random.rand(3)
    bboxes.append([x0,x1,y0,y1])

plt.subplot(211)
draw_img(img, bboxes)

# apply transformation
if random.random() < 0.2:
    shear = random.random() * 30 - 15        # -15 deg to +15 deg
    shear = np.deg2rad(shear)
else:
    shear = 0
if random.random() < 0.2:
    rotate = random.random() * 30 - 15       # -15 deg to +15 deg
    rotate = np.deg2rad(rotate)
else:
    rotate = 0
if random.random() < 0.2:
    scale = [random.random() * 0.3 + 0.85, random.random() * 0.3 + 0.85]     # 0.85 to 1.15
else:
    scale = [1.0, 1.0]

translation = [img.shape[0] * 0.5, img.shape[1] * 0.5]

if random.random() < 0.2:
    img = img[:,::-1,:]
    for b in bboxes:
        b[2] = w - b[2]
        b[3] = w - b[3]

img = color.rgb2hsv(img)
if random.random() < 0.2:
    # brightness
    img[:,:,2] += random.random() * 0.3 - 0.15  # -0.15 to + 0.15
    img = img % 1
if random.random() < 0.2:
    # hue
    img[:,:,0] += random.random() * 0.3 - 0.15  # -0.15 to + 0.15
    img = img % 1
img = color.hsv2rgb(img)

transformation = transform.AffineTransform(shear=shear, scale=scale, rotation=rotate, translation=translation)
mat = transformation.params

img = (img * 255).astype(np.uint8)
img = transform.warp(img, transformation.inverse, order=0, output_shape=(img.shape[0] * 3, img.shape[1] * 3))

bboxes = np.array(bboxes)
bboxes2 = np.zeros([bboxes.shape[0], 4, 2])
bboxes2[:,:2,0] = bboxes[:,[0]]
bboxes2[:,2:,0] = bboxes[:,[1]]
bboxes2[:,::2,1] = bboxes[:,[2]]
bboxes2[:,1::2,1] = bboxes[:,[3]]
bboxes_transformed = np.zeros(bboxes2.shape)
bboxes_transformed[:,:,1] = bboxes2[:,:,1] * mat[0,0] + bboxes2[:,:,0] * mat[0,1] + mat[0,2] # X = a0*x + a1*y + a2
bboxes_transformed[:,:,0] = bboxes2[:,:,1] * mat[1,0] + bboxes2[:,:,0] * mat[1,1] + mat[1,2] # Y = b0*x + b1*y + b2
bboxes_transformed2 = np.zeros([bboxes.shape[0], 4])
bboxes_transformed2[:,0] = np.min(bboxes_transformed[:,:,0], axis=1)
bboxes_transformed2[:,1] = np.max(bboxes_transformed[:,:,0], axis=1)
bboxes_transformed2[:,2] = np.min(bboxes_transformed[:,:,1], axis=1)
bboxes_transformed2[:,3] = np.max(bboxes_transformed[:,:,1], axis=1)


plt.subplot(212)
draw_img(img, bboxes_transformed2)
plt.show()