import cv2
import numpy as np
import matplotlib.pyplot as plt
import math




print("Digital Image Processing Homework")


# 1.Read The image
img1 = cv2.imread('img.jpg', 1)

window_name='Original Image'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name,img1)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 2.Convert To GrayScale
img_gray= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
window_name='Grayscale Conversion OpenCV'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name,img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 3.Show Histogram
plt.subplot(1,2,1)
plt.imshow(img_gray,cmap='gray')
plt.title('image')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
histogram=plt.hist(img_gray.ravel(),256,[0,255])
plt.title('histogram')
plt.show()


# 4.Features
# get mean from image
imean = np.mean(img_gray)
print("Mean: ",imean)
# get standard deviation from image
grayf = img_gray.astype(np.float32)
grayf2 = grayf * grayf
imeanf2 = np.mean(grayf2)
ivar = imeanf2 - imean**2
istd = math.sqrt(ivar)
print('Standard Deviation: ', istd)
# get median from image
imedian=np.median(img_gray)
print("Median: ",imedian)
#Mode
arr = []
arr=[0 for x in range(256)]

for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        ind=img_gray[i,j]
        arr[ind]=arr[ind]+1
ma=max(arr)
mode=arr.index(ma)
print("Mode: ",mode)


#5 a.the intensity transformation functions

#Global grayscale linear transformation
def global_linear_transmation(img):
    img=1-img
    return img

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_gray,cmap='gray')
plt.title('original image')
plt.subplot(1,2,2)
image1=global_linear_transmation(img_gray)
plt.imshow(image1,cmap='gray')
plt.title('After linear transformation')
plt.show()

# Log Transformation
c = 255 / np.log(1 + np.max(img_gray))
log_image = c * (np.log(img_gray + 1))


# float value will be converted to int according to the dtype
log_image = np.array(log_image, dtype=np.uint8)
plt.title('Log Transformation')
plt.imshow(log_image,cmap="gray")
plt.show()

# Power Law
for gamma in [0.1, 0.5, 1.2, 2.2]:

    gamma_corrected = np.array(255 * (img_gray / 255) ** gamma, dtype='uint8')
    plt.title('Power Law with gamma =' + gamma.__str__())
    plt.imshow(gamma_corrected, cmap="gray")
    plt.show()

# Piecewise-Linear Transformation
# Function to map each intensity level to output intensity level.
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

r1 = 70
s1 = 0
r2 = 140
s2 = 255

# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)

# contrast stretching.
contrast_stretched = pixelVal_vec(img_gray, r1, s1, r2, s2)
cv2.imwrite('contrast_stretch.jpg', contrast_stretched)
plt.title("Piecewise-Linear Transformation")
plt.imshow(contrast_stretched,cmap="gray")
plt.show()

#5. b. Equalization
equ = cv2.equalizeHist(img_gray)

plt.title("Histogram Equalization")
plt.imshow(equ,cmap="gray")
plt.show()

