# 🎨 AI Color Palette Extractor (K-Means Clustering)

An intelligent Machine Learning tool designed to analyze images and extract professional color palettes. This tool helps creatives find the dominant aesthetic of any visual asset instantly.

## 🔗 Live Demo
Check out the live app here: [https://ai-colour-palette-extractor-q9bj6c9hdlxqkavih9572d.streamlit.app/](https://ai-face-blur-privacy-filter-fohbwgxspywoewxhwzetcn.streamlit.app/)

## 🚀 Project Overview
As a **Creative Technologist and Animator**, I built this tool to bridge the gap between Data Science and Design. Choosing the right color scheme is essential in high-end animation and video production. This application automates that process using **Unsupervised Machine Learning**.

### Key Features:
* **AI-Powered Extraction:** Uses the **K-Means Clustering** algorithm to mathematically group pixels into dominant color clusters.
* **Dynamic Palette Selection:** Allows users to choose between 2 to 10 dominant colors depending on the design requirement.
* **Instant Hex Codes:** Automatically generates **Hex codes** for every extracted color, ready to be used in software like After Effects, Photoshop, or Blender.
* **Visual Insights:** Displays the original image alongside the AI-generated palette for side-by-side comparison.



## 🛠️ Tech Stack
* **Language:** Python 3.12
* **ML Library:** Scikit-learn (K-Means)
* **Image Processing:** PIL (Pillow), NumPy
* **Frontend:** Streamlit
* **Visualization:** Matplotlib

## 🧠 The Logic
The system processes the image through a standard Machine Learning pipeline:
1. **Data Preparation:** Converting image pixels into a 3D RGB color space.
2. **Clustering:** Iteratively grouping similar pixels together.
3. **Centroid Calculation:** Finding the "average" color for each group to represent the final palette.



---
Developed by **Rukshan Weerasekara** *Creative Technologist | Animator | AI Developer*
