import tempfile
from KalmanFilter import KalmanFilter
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
import cv2 
import re
import os



os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
cap = cv2.VideoCapture('./carInWay.mp4')
car_searched = "85040"

def car_detect(cap):
    # Initialiser OCR et modèle de détection
    ocr = PaddleOCR(lang="en", use_angle_cls=True, use_gpu=False)
    LPD_model = YOLO("./LisencePlateDetector/weights/best.pt")
    kalman = KalmanFilter()

    # Couleurs pour l'affichage
    colors = {
        'actual': (0, 255, 0),  # Vert pour la position réelle
        'predicted': (0, 0, 255)  # Rouge pour la prédiction
    }

    # Charger la vidéo d'entrée
#     cap = cv2.VideoCapture(input_video_path)

    # Obtenir les propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4

    def paddle_ocr(frame, x1, y1, x2, y2):
        """ Effectue l'OCR sur la plaque détectée """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_frame = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        result = ocr.ocr(cropped_frame, det=False, rec=True, cls=False)

        text = ""
        for r in result:
            detected_text, confidence = r[0][0], r[0][1]
            if confidence > 0.5:  # Confiance minimale de 50 %
                text = detected_text
        text = re.sub(r'\W', '', text)  # Supprimer les caractères spéciaux
        text = text.replace("O", "0").replace(" ", "")  # Corriger les erreurs fréquentes
        return text

    # Lire les frames
    frame_nmr = -1
    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Détection de plaques
        LP_cars = LPD_model(frame)[0]
        for LP_car in LP_cars.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, score, class_id = LP_car
            label = paddle_ocr(frame, xcar1, ycar1, xcar2, ycar2)
            x = (xcar1 + xcar2) / 2
            y = (ycar1 + ycar2) / 2

            if not kalman.is_initialized:
                kalman.initialize(x, y)
            else:
                kalman.correct(x, y)

            # Afficher le texte détecté
            cv2.putText(frame, label, (int(xcar1), int(ycar1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Dessiner la boîte autour de la plaque détectée
            cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), colors['actual'], 2)

        # Suivi Kalman
        if kalman.is_initialized:
            predicted_x, predicted_y = kalman.predict()
            cv2.circle(frame, (int(predicted_x), int(predicted_y)), 5, colors['predicted'], -1)

        # Ajouter la frame modifiée à la vidéo de sortie
        out.write(frame)

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path


if __name__ == "__main__":
     car_detect(cap)
