import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from predictionModel import RustDetection
import os

def load_image(image_file):
    img = Image.open(image_file)
    return img

def draw_image_with_boxes(image_file,pre=0,confidence_threshold=0.5,row=1,col=1):
    with Image.open(image_file) as img:
        draw=ImageDraw.Draw(img)
        draw.line(((5,10),(20,10)),fill='red',width=5)
    return img

def main():
    # to do list
    # - input image, output "find a rust"
    # - deploy in the cloud
    # - zzlab webpage
    # - standard
    # - single image model and show like the ROOSTER
    # - multi images model and show multiple images in map
    st.sidebar.title("RustNet")
    image_files = st.sidebar.file_uploader("Upload Images", type=["png","jpg","jpeg"], accept_multiple_files = True)
    # weight_file = st.sidebar.file_uploader("Upload weight",type=['pth'])
    weight_file = './RustNet.pth'
    row_num = st.sidebar.number_input('Row', min_value=0)
    col_num = st.sidebar.number_input('Col', min_value=0)
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

    if image_files is not None and weight_file is not None:
        dlinput = {}
        dlinput['row'] = row_num
        dlinput['col'] = col_num
        dlinput['weight'] = weight_file
        dlinput['model'] = "resnet18"

        for img in image_files:
            dlinput['imagepath'] = img
            st.image(load_image(img))
            pre = RustDetection(dlinput=dlinput)
            re_rust = [x for x in pre if x > confidence_threshold]
            if len(re_rust) > 0:
                st.text("Detect Stripe rust!")
                st.dataframe(re_rust)
            else:
                st.text("No Stripe rust found!")
            # st.image(img_draw(image_files[0]))
            # st.dataframe(pre)

        # if len(image_files) == 1:
            # st.image(load_image(image_files[0]))
            # st.image(img_draw(img,pre,confidence_threshold,row_num,col_num))


if __name__ == '__main__':
    main()