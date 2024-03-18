patchSize = 3 # patch size this is the radis the size is D*2+1

"""Read two image and transform them to gray"""

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import math

import cv2
from PIL import Image

# Read the two image and normalize the pixel value
imageLeft = plt.imread("conesLeft.jpg")/255
imageRight = plt.imread("conesRight.jpg")/255

# Convert the color image into grayscale image
imageLeftGray = rgb2gray(imageLeft)
imageRightGray = rgb2gray(imageRight)

# Get the size of the image
nrows = imageLeft.shape[0]
ncolns = imageLeft.shape[1]

# Generate the depth image that has the same size with the orignal

imageDepth = np.zeros((nrows, ncolns))

new_imleftgrey = np.zeros((nrows+2*patchSize,ncolns+2*patchSize))
new_imrightgrey = np.zeros((nrows+2*patchSize,ncolns+2*patchSize))


new_imleftgrey[patchSize:patchSize+nrows, patchSize:patchSize+ncolns] = imageLeftGray
new_imrightgrey[patchSize:patchSize+nrows,patchSize:patchSize+ncolns] = imageRightGray

def normalizeMatrixCol(matrix, size):
  nrows = matrix.shape[0]
  ncols = matrix.shape[1]
  size = size[0]

  nXm = np.zeros(((size - 1) * size, ncols))

  for i in range(0, ncols - size):
    patch = matrix[0:size, i:i+size]
    nXm[:, i] = ((patch - np.mean(patch)) / patch.std()).flatten()

  return nXm

# Main loop that repeat nrows time

for x in range(0, nrows):
  i = x + patchSize
  print("Starting row %s" % x)

  DSIImage = np.zeros((ncolns, ncolns))

# Generate a patch for NCC operation
  bigPatch = new_imleftgrey[i-patchSize:i+patchSize, 1:ncolns + 2 * patchSize]
  fMatrix = normalizeMatrixCol(bigPatch, (2*patchSize+1,2*patchSize+1))

  bigPatch = new_imrightgrey[i-patchSize:i+patchSize, 1:ncolns + 2 * patchSize]
  gMatrix = normalizeMatrixCol(bigPatch, (2*patchSize+1,2*patchSize+1))

  DSIImage = 1 - np.matmul(np.transpose(gMatrix), fMatrix)

  print("Finish NCC")

  # # Dynamic Program

  dynamicGraph = np.zeros((ncolns, ncolns))
  directMap = np.zeros((ncolns, ncolns))
  pathMap = np.zeros((ncolns, ncolns))
  occlusion = 0.1

  # # Init the upper edge and the left edge

  for pt in range(1, ncolns):
    dynamicGraph[pt,0] = dynamicGraph[pt,1] + occlusion
    directMap[pt,0] = 2
    dynamicGraph[0,pt] = dynamicGraph[1,pt] + occlusion
    directMap[0,pt] = 3

  # # Dynamic programing step 1:
  for pti in range(1,ncolns):
        for ptj in range(1,ncolns):
            min1 = dynamicGraph[pti-1,ptj-1] + DSIImage[pti, ptj] # diagnals path
            min2 = dynamicGraph[pti-1,ptj] + occlusion     # vertical path
            min3 = dynamicGraph[pti,ptj-1] + occlusion     # horizotal path
            dynamicGraph[pti,ptj] = min(min1,min2,min3)
            minValue =  min(min1,min2,min3)
            if minValue == min1:
              directMap[pti,ptj] = 1
            elif minValue == min2:
              directMap[pti,ptj] = 2
            elif minValue == min3:
              directMap[pti,ptj] = 3

  # # Dynamic programing step 2
  p = ncolns - 1
  q = ncolns - 1

  while p > 1 or q > 1:
        pathMap[p,q] = 1
        if directMap[p,q] == 1:
          p = p - 1
          q = q - 1
          imageDepth[x,q] = abs(q-p)
        elif directMap[p,q] == 2:
          p = p - 1
        elif directMap[p,q] == 3:
          q = q - 1
          imageDepth[x,q] = abs(q-p)

# Display the depth image
plt.imshow(imageDepth, plt.cm.gray)
# Display the left image and right image.
# plt.imshow(imageRight)
# plt.imshow(imageLeft)
