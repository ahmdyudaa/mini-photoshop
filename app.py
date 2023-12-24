import cv2 as cv
from scipy import ndimage
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from image_processing import convert_to_gray, convert_to_binary, convert_to_negative, convert_to_smooth, detect_edge, change_brightness, equalization, rotate, flip, contrast, sharpness


st.set_option('deprecation.showPyplotGlobalUse', False)

#Histogram
def display_histogram(img):
    # Calculate histogram
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    # Plot histogram
    plt.plot(hist)
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    st.pyplot()

st.image('assets/illustration design.svg')
# UI
st.title("Mini Photoshop")
st.sidebar.header("Fitur")

# Upload image
upload_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"])

if upload_image is not None:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
    img2 = cv.imdecode(file_bytes, 2)

    # Sidebar
    menu = st.sidebar.selectbox("Pilih Menu Image Processing", 
    ["Grayscale", "Binary", "Negative", "Edge Detection", "Smoothing", "Brightness", "Equalization", "Rotate", "Flip" , "Contrast", "Sharpness"])

    # Main
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(img, channels="BGR", use_column_width=True)
        display_histogram(img)
        

    with col2:
        st.header("Edited Image")

        if menu == 'Grayscale':
            img_gray = convert_to_gray(img)
            st.image(img_gray, use_column_width=True)
            display_histogram(img_gray)

        elif menu == 'Binary':
            treshold = st.sidebar.slider(
                    "Treshold",
                    value=128,
                    min_value=0,
                    max_value=255,
                    )
            img_binary  = convert_to_binary(img2, treshold)
            st.image(img_binary, use_column_width=True)
            display_histogram(img_binary)

        elif menu == 'Negative':
            img_negative  = convert_to_negative(img)
            st.image(img_negative, use_column_width=True)
            display_histogram(img_negative)

        elif menu == 'Edge Detection':
            method = st.sidebar.radio(
            "Pilih Metode Konvolusi",
            ["Canny", "Sobel", "Robert", "Prewit"])
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_gaussian = cv.GaussianBlur(img_gray, (1,1),0)
            img_edge  = detect_edge(img_gaussian, method)
            
            st.image(img_edge, use_column_width=True)
            display_histogram(img_edge)
            

        elif menu == 'Smoothing':
            factor = st.sidebar.slider(
                    "Factor",
                    value=5,
                    min_value=1,
                    max_value=19,
                    step=2,
                    )
            img_smoothing = convert_to_smooth(img, factor)
            img_smoothing_rgb = cv.cvtColor(img_smoothing, cv.COLOR_BGR2RGB)
            st.image(img_smoothing_rgb, use_column_width=True)
            
            display_histogram(img_smoothing_rgb)

        elif menu == 'Brightness':
            factor = st.sidebar.slider(
                    "Brightness Factor",
                    value=5,
                    min_value=1,
                    max_value=10,
                    )
            img_brightness = change_brightness(img, factor)
            img_brightness_rgb = cv.cvtColor(img_brightness, cv.COLOR_BGR2RGB)
            st.image(img_brightness_rgb, use_column_width=True)
            display_histogram(img_brightness_rgb)

        elif menu == "Equalization":
            img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])
            img_equalize_rgb = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)
            st.image(img_equalize_rgb,use_column_width=True)
            display_histogram(img_equalize_rgb)


        elif menu == "Rotate":
            rotate_degree = st.sidebar.number_input('Input Derajat Rotasi', min_value=0, max_value=360)
            img_rotate = rotate(img, rotate_degree)
            img_rotate_rgb = cv.cvtColor(np.array(img_rotate), cv.COLOR_BGR2RGB)
            st.image(img_rotate_rgb, use_column_width=True)
            display_histogram(img_rotate_rgb)

        elif menu == "Flip":
            arrow = st.sidebar.radio(
            "Pilih Arah Flip",
            ["Horizonal", "Vertikal", "Both"])
            img_flip = flip(img, arrow)
            img_flip_rgb = cv.cvtColor(np.array(img_flip), cv.COLOR_BGR2RGB)
            st.image(img_flip_rgb, use_column_width=True)
            display_histogram(img)
        
        elif menu == "Contrast":
            factor = st.sidebar.number_input('Factor', min_value=1, max_value=10)
            img_contrast = contrast(img, factor)
            img_contrast_rgb = cv.cvtColor(np.array(img_contrast), cv.COLOR_BGR2RGB)
            st.image(img_contrast_rgb, use_column_width=True)
            display_histogram(img_contrast_rgb)
        
        elif menu == "Sharpness":
            factor = st.sidebar.number_input('Factor', min_value=1, max_value=10)
            img_sharpness = sharpness(img, factor)
            img_sharpness_rgb = cv.cvtColor(np.array(img_sharpness), cv.COLOR_BGR2RGB)
            st.image(img_sharpness_rgb, use_column_width=True)
            display_histogram(img_sharpness_rgb)

else:
    st.warning("Silakan pilih gambar untuk memulai pengolahan.")


cpy =('''
    <style>
        #MainMenu{
            display: none;
        }
        footer{
            display: none;
        }
    </style>
'''
)
st.markdown(cpy, unsafe_allow_html=True)