import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

#Page Configuration 
st.set_page_config(page_title="AI Color Palette", layout="centered")

st.title("🎨 AI Color Palette Extractor")
st.markdown("Upload an image to extract its dominant color palette using Machine Learning.")

# File Uploader 
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    
    num_colors = st.slider("Select number of colors", 2, 10, 5)
    
    if st.button("Extract Palette"):
        with st.spinner("AI is analyzing colors..."):
            # Preprocess the image
            img_np = np.array(img.convert('RGB'))
            img_flat = img_np.reshape(-1, 3) # Flatten into (pixels, 3)
            
            # Apply K-Means Clustering
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(img_flat)
            
            # Get the dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            # Display the color palette
            st.subheader("Extracted Palette")
            cols = st.columns(num_colors)
            
            for i, color in enumerate(colors):
                # Convert RGB to Hex format
                hex_code = '#%02x%02x%02x' % tuple(color)
                
                # Display color box and hex code
                cols[i].markdown(f'<div style="background-color:{hex_code}; height:100px; border-radius:10px;"></div>', unsafe_allow_html=True)
                cols[i].code(hex_code)

st.markdown("---")
st.caption("AI-Powered Color Extraction Tool")
