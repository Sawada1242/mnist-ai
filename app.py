import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model("mnist_model.keras")

st.title("手書き数字AI（MNIST）")

file = st.file_uploader("数字画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if file is not None:
    img = Image.open(file).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)

    pred = model.predict(img_array)

    st.image(img, caption="入力画像", width=150)
    st.write("予測結果:", np.argmax(pred))