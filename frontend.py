import streamlit as st
import requests
import time
import base64
import os


# Interface Streamlit
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()
# ajout de l'image au background
img_base64 = get_base64("car_1.jfif")  

page_bg_img = f"""
<style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
      }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("# Détection de véhicules et reconnaissance de plaques")

# telecharger la video 
uploaded_file = st.file_uploader("charger la video ici", type=["mp4", "mpeg4"])
done = False
if uploaded_file :
    st.video(uploaded_file)
    st.subheader("Vidéo avec Détection de Véhicules")
    # Lire le fichier en mémoire et l'envoyer à FastAPI
    files = {"file": ("video.mp4", uploaded_file.getvalue(), "video/mp4")}
    response = requests.post("http://127.0.0.1:8000/process_video",
                                 files=files)
    if response.text.strip(): #it wants print the output until the backend finishes its work
        done = True

if done:
    time.sleep(5)
    output_path = response.text.strip().strip('"')
    st.write(f"Output path: {output_path}")  # Debugging information

    # Check if the file exists before displaying it
    if os.path.exists(output_path):
        st.video(output_path)
    else:
        st.error("Le fichier vidéo traité n'existe pas.")

 