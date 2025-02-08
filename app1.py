import streamlit as st
import cv2
import time
import tempfile
import os
import numpy as np
import re
from KalmanFilter import KalmanFilter
from ultralytics import YOLOv10
from paddleocr import PaddleOCR

# Charger les modèles YOLO et PaddleOCR
model = YOLOv10("./weights/best.pt")  # Remplacez par le chemin de votre modèle YOLO
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Fonction OCR pour détecter le texte sur une plaque
def paddle_ocr(frame, x1, y1, x2, y2):
    frame_cropped = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame_cropped, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        score = r[0][1]
        score = 0 if np.isnan(score) else int(score * 100)
        if score > 60:
            text = r[0][0]
    pattern = re.compile(r"[\W]")
    text = pattern.sub("", text).replace("O", "0").upper()
    return text

# Fonction principale de traitement vidéo
def process_video(video_path, car_plate):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Erreur lors de l'ouverture de la vidéo.")
        return None

    # Obtenir les paramètres de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Créer un fichier temporaire pour la sortie vidéo
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_output.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialiser le filtre de Kalman
    kalman = KalmanFilter()

    # Chronométrage
    start_time = time.time()

    # Traitement image par image
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Vérifier si le temps écoulé dépasse 100 secondes
        elapsed_time = time.time() - start_time
        if elapsed_time >= 100:
            st.warning("⏳ Temps limite atteint (100 sec), arrêt du traitement.")
            break

        # Détection des plaques avec YOLO
        results = model.predict(frame, conf=0.45)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_plate = paddle_ocr(frame, x1, y1, x2, y2)

                # Comparaison avec la plaque recherchée
                def compare_plates(detected, target):
                    return detected[:3] == target[:3] and detected[-1] == target[-1]

                if compare_plates(detected_plate, car_plate):
                    # Dessiner un rectangle et afficher le texte
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, detected_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Suivi avec Kalman
                    if not kalman.is_initialized:
                        kalman.initialize((x1 + x2) / 2, (y1 + y2) / 2)
                    else:
                        kalman.correct((x1 + x2) / 2, (y1 + y2) / 2)

                    # Afficher la position prévue par Kalman
                    predicted_x, predicted_y = kalman.predict()
                    cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, (255, 0, 0), -1)

        # Ajouter l'image modifiée à la vidéo de sortie
        out.write(frame)

    # Libération des ressources
    cap.release()
    out.release()

    return output_path

# Interface Streamlit
st.title("🚗 Suivi de plaques avec YOLO, OCR et Kalman Filter")
st.write("Détection et suivi de plaques avec arrêt automatique après 100 secondes.")

# Entrée utilisateur
car_plate = st.text_input("🔎 Entrez la plaque recherchée :", "")
uploaded_video = st.file_uploader("📂 Téléchargez une vidéo :", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.video(uploaded_video)

if uploaded_video and car_plate:
    # Sauvegarder la vidéo temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

    # Lancer le traitement
    st.write("🛠️ Traitement en cours...")
    with st.spinner("Analyse de la vidéo... ⏳"):
        output_video_path = process_video(video_path, car_plate)

    # Afficher le résultat
    if output_video_path and os.path.exists(output_video_path):
        st.success("✅ Analyse terminée ! Voici la vidéo traitée :")
        st.video(output_video_path)
        st.download_button(
            label="📥 Télécharger la vidéo traitée",
            data=open(output_video_path, "rb"),
            file_name="output_video.mp4",
            mime="video/mp4",
        )
    else:
        st.error("Erreur : La vidéo traitée n'a pas été trouvée ou n'existe pas.")

    # Ajoutez des messages de débogage pour vérifier le chemin du fichier
    st.write(f"Chemin de la vidéo traitée : {output_video_path}")
    st.write(f"Existence du fichier : {os.path.exists(output_video_path)}")
