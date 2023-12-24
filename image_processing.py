import cv2 as cv
import numpy as np
from skimage import filters, color
from scipy import ndimage
from PIL import Image, ImageEnhance

#Function Image Processing

#grayscale
def convert_to_gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

#detect edge
def detect_edge(img, method):
    if method == "Canny":
        edges = cv.Canny(img, 100, 200)
    if method == "Sobel":
        img_sobel_x = cv.filter2D(img, cv.CV_8U, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        img_sobel_y = cv.filter2D(img, cv.CV_8U, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
        # result = np.sqrt(img_sobel_x * img_sobel_x + img_sobel_y*img_sobel_y)
        edges = img_sobel_x + img_sobel_y
    if method == "Prewit":
        edgesx = cv.filter2D(img, cv.CV_8U, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        edgesy = cv.filter2D(img, cv.CV_8U, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        edges = edgesx + edgesy
    if method == "Robert":
        edgesx = cv.filter2D(img, cv.CV_8U, np.array([[1, 0], [0, -1]]))
        edgesy = cv.filter2D(img, cv.CV_8U, np.array([[0, 1], [-1, 0]]))
        edges = edgesx + edgesy

    return edges
    
#negative
def convert_to_negative(img):
    negative = cv.bitwise_not(img)
    return negative

#binary
def convert_to_binary(img, treshold):
    ret, binary = cv.threshold(img, treshold, 255, cv.THRESH_BINARY)
    return binary

#smooth
def convert_to_smooth(img, factor):
    smooth = cv.GaussianBlur(img, (factor, factor), 0) 
    return smooth

#brightness
def change_brightness(img, factor):
    enhancer = ImageEnhance.Brightness(Image.fromarray(img).convert('RGB'))
    brightened_image = np.asarray(enhancer.enhance(factor)).astype(np.uint8)
    return brightened_image

#eqalization
def equalization(img):
    equalization_image = cv.equalizeHist(img)
    return equalization_image

#Rotate
def rotate(img, *argv):
    rotated_image = ndimage.rotate(img, *argv, reshape=False)
    rot_img = Image.fromarray(rotated_image)
    return rot_img

#Flip
def flip(img, arrow):
    if arrow == "Horizonal":
        flip_image = cv.flip(img, 1)
    if arrow == "Vertikal":
        flip_image = cv.flip(img, 0)
    if arrow == "Both":
        flip_image = cv.flip(img, -1)

    flip_img = Image.fromarray(flip_image)
    return flip_img

#Image Enhance
def contrast(img, factor):
    enhancer = ImageEnhance.Contrast(Image.fromarray(img).convert('RGB'))
    contrast_img = np.asarray(enhancer.enhance(factor)).astype(np.uint8)
    return contrast_img

def sharpness(img, factor):
    enhancer = ImageEnhance.Sharpness(Image.fromarray(img).convert('RGB'))
    sharpness_img = np.asarray(enhancer.enhance(factor)).astype(np.uint8)
    return sharpness_img