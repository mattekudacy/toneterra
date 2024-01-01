import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_color(image, number_of_colors, show_chart):
    modified_image = cv.resize(image, (600, 400), interpolation = cv.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]

    if (show_chart):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(counts.values(), labels=hex_colors, colors=hex_colors)
        st.pyplot(fig)

    return hex_colors

def main():
    st.title('ToneTerra🌈')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)  
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        number_of_colors = st.slider("Choose number of colors to generate", 1, 20, 5)

        show_chart = st.checkbox("Show Chart")

        if st.button("Generate Color Palette"):
            colors = get_color(image, number_of_colors, show_chart)
            st.write("Generated Color Palette:")
            for color in colors:
                st.write(color)

if __name__ == "__main__":
    main()
