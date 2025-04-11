import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# --- Page title ---
st.set_page_config(page_title="Image Transformer", layout="wide")
st.title("Image Transformation App")
st.markdown("Upload an image, choose transformations, and view the result in real time!")

# --- Upload ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# --- Sidebar Transform Options ---
st.sidebar.header("🔧 Transformations")
apply_gray = st.sidebar.checkbox("Convert to Grayscale")
apply_contrast = st.sidebar.checkbox("Contrast Stretching")
apply_negative = st.sidebar.checkbox("Image Negative")
apply_scale = st.sidebar.checkbox("Resize Image")
apply_flip_horizontal = st.sidebar.checkbox("Flip Horizontally")
apply_flip_vertical = st.sidebar.checkbox("Flip Vertically")
apply_blur = st.sidebar.checkbox("Apply Blur")

if apply_scale:
    fx = st.sidebar.slider("Scale X", 0.1, 2.0, 0.5)
    fy = st.sidebar.slider("Scale Y", 0.1, 2.0, 0.5)

if apply_blur:
    blur_ksize = st.sidebar.slider("Blur Intensity (Odd number)", 1, 51, 5, step=2)

# --- Contrast Stretching Function ---
def contrast_stretch(img):
    if len(img.shape) == 2:  # Grayscale
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            return img.copy()
        stretched = np.clip((img - min_val) * (255.0 / (max_val - min_val)), 0, 255)
        return stretched.astype(np.uint8)
    else:  # Color
        stretched = np.zeros_like(img)
        for i in range(3):
            channel = img[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val == min_val:
                stretched[:, :, i] = channel
            else:
                stretched[:, :, i] = np.clip((channel - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)
        return stretched

# --- Process Image ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB for consistency
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    transformed = image_cv.copy()

    # Apply transformations
    if apply_gray:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    if apply_contrast:
        transformed = contrast_stretch(transformed)

    if apply_negative:
        transformed = cv2.bitwise_not(transformed)

    if apply_scale:
        transformed = cv2.resize(transformed, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    if apply_flip_horizontal:
        transformed = cv2.flip(transformed, 1)

    if apply_flip_vertical:
        transformed = cv2.flip(transformed, 0)

    if apply_blur:
        transformed = cv2.GaussianBlur(transformed, (blur_ksize, blur_ksize), 0)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Transformed Image")
        if len(transformed.shape) == 2:
            st.image(transformed, channels="GRAY", use_container_width=True)
            image_to_save = Image.fromarray(transformed)
        else:
            transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
            st.image(transformed_rgb, use_container_width=True)
            image_to_save = Image.fromarray(transformed_rgb)

        # Download button
        buf = BytesIO()
        image_to_save.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Transformed Image",
            data=byte_im,
            file_name="transformed_image.png",
            mime="image/png"
        )
else:
    st.info("⬆️ Upload an image to get started.")
