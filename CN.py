
from PIL import Image,ImageOps
import numpy as np
import streamlit as st 
import tensorflow as tf
import os
from tensorflow.keras import Sequential

from tensorflow.keras.models import load_model

from io import BytesIO




MODEL= load_model("pest.h5")

file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

st.title("Detection of agricultural insect pests using the cnn algorithm")


	
	  





def predict(image_D):
	
	size=(256,256)
	image=ImageOps.fit(image_D,size,Image.ANTIALIAS)
	img=np.asarray(image)
	img_reshape=img[np.newaxis,...]

	prediction = MODEL.predict(img_reshape)
	return prediction

if file is None:
	st.text("pleas upload img ")
else:
	image=Image.open(file)
	st.image(image,use_column_width=True)
	st.subheader("Image")
	result=predict(image)
	CLASS_NAMES=['BA', 'HA', 'MP', 'SE', 'SL', 'TP', 'TU', 'ZC']
	image_class = CLASS_NAMES[np.argmax(result)]
	predictions = f"img is {image_class}, accurcy {np.max(result):22f}"
		# RE=st.write(result)
	st.success(predictions)



