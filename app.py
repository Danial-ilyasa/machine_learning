import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from pathlib import Path
from sistem import JaringanSyarafTiruan as JST
from PIL import Image
import pandas as pd

MODEL_FILE = Path(__file__).resolve().with_name("model.pkl")
model = JST.muat(MODEL_FILE)

def flatten_sample(image):
    image = np.array(image, dtype=np.float32)
    return image.reshape(-1, 1)


st.title("Aplikasi Jaringan Syaraf Tiruan untuk Klasifikasi Angka")
k1, k2 = st.columns([3, 3])
with k1:
    CANVAS_SIZE = 280
    canvas = st_canvas(
            fill_color="#FFFEFE",
            stroke_width=13,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True,
          )
with k2:
    
    if canvas.image_data is not None:
        true_img = canvas.image_data[::-1, ::-1, :3]

        if np.sum(true_img) < 1000:
            st.write("Gambar terlalu kosong, silakan menggambar lebih banyak.")
            st.stop()
        else:
            img = np.array(canvas.image_data)
            pil_img = Image.fromarray(img.astype("uint8"))
            pil_img = pil_img.convert("L")
            pil_img = pil_img.resize((28, 28))

            # Tambahkan di app.py sebelum prediksi
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            img_array = flatten_sample(img_array)
            x = model.umpan_maju(img_array)
            st.write(f"Prediksi: {np.argmax(x)}")

        data = pd.DataFrame(x, columns=["Probabilitas"])
        st.bar_chart(data, width=300, height=200)

    else:
        st.write("Silakan menggambar di kanvas untuk mendapatkan prediksi.")
    
    if MODEL_FILE.exists():
        st.write("Model berhasil dimuat dari", MODEL_FILE)


