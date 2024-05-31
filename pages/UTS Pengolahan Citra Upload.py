import streamlit as st
import cv2
import numpy as np

def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image

def calculate_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = 1 + contrast / 127
    beta = brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def find_contours(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray_image, 127, 255, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    background_removed = cv2.bitwise_and(image, image, mask=mask)
    return background_removed

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cx, cy) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def main():
    st.set_page_config(layout="wide")
    st.title("Memanipulasi Gambar")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.sidebar.subheader("Image Manipulations")
            selected_option = st.sidebar.selectbox("Select an option", ["RGB to HSV", "Histogram", "Brightness and Contrast", "Contour", "Grayscale", "Blur", "Edge Detection", "Thresholding", "Rotate", "Resize", "Flip", "Crop", "Remove Background"], index=None)

            if selected_option == "RGB to HSV":
                hsv_image = rgb_to_hsv(image)
                st.image(hsv_image, caption="HSV Image", use_column_width=True)

            elif selected_option == "Histogram":
                histogram = calculate_histogram(image)
                st.bar_chart(histogram)

            elif selected_option == "Brightness and Contrast":
                brightness = st.slider("Brightness", -100, 100, 0)
                contrast = st.slider("Contrast", -100, 100, 0)
                adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
                st.image(adjusted_image, caption="Adjusted Image", use_column_width=True)

            elif selected_option == "Contour":
                contours = find_contours(image)
                image_with_contours = np.copy(image)
                cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)
                st.image(image_with_contours, caption="Image with Contours", use_column_width=True)

            elif selected_option == "Grayscale":
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

            elif selected_option == "Blur":
                blur_amount = st.slider("Blur Amount", 1, 10, 1)
                blurred_image = cv2.GaussianBlur(image, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
                st.image(blurred_image, caption="Blurred Image", use_column_width=True)

            elif selected_option == "Edge Detection":
                edges = cv2.Canny(image, 100, 200)
                st.image(edges, caption="Edges Detected", use_column_width=True)

            elif selected_option == "Thresholding":
                _, thresh_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY)
                st.image(thresh_image, caption="Threshold Image", use_column_width=True)

            elif selected_option == "Rotate":
                angle = st.slider("Angle", 0, 360, 0)
                rotated_image = rotate_image(image, angle)
                st.image(rotated_image, caption="Rotated Image", use_column_width=True)

            elif selected_option == "Resize":
                width = st.number_input("Width", min_value=1, value=image.shape[1])
                height = st.number_input("Height", min_value=1, value=image.shape[0])
                resized_image = cv2.resize(image, (int(width), int(height)))
                st.image(resized_image, caption="Resized Image", use_column_width=True)

            elif selected_option == "Flip":
                flip_code = st.selectbox("Flip Axis", ["Horizontal", "Vertical"], index=0)
                flipped_image = cv2.flip(image, 0 if flip_code == "Vertical" else 1)
                st.image(flipped_image, caption="Flipped Image", use_column_width=True)

            elif selected_option == "Crop":
                x_start = st.slider("X Start", 0, image.shape[1], 0)
                y_start = st.slider("Y Start", 0, image.shape[0], 0)
                x_end = st.slider("X End", x_start, image.shape[1], image.shape[1])
                y_end = st.slider("Y End", y_start, image.shape[0], image.shape[0])
                if x_end > x_start and y_end > y_start:
                    cropped_image = image[y_start:y_end, x_start:x_end]
                    st.image(cropped_image, caption="Cropped Image", use_column_width=True)
                else:
                    st.error("End values must be greater than start values for cropping.")

            elif selected_option == "Remove Background":
                background_removed_image = remove_background(image)
                st.image(background_removed_image, caption="Background Removed", use_column_width=True)

            st.subheader("Hasil Manipulasi Gambar")

if __name__ == "__main__":
    main()
