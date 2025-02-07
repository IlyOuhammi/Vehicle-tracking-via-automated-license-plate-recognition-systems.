from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from Car_detect import car_detect  # Importer la fonction car_detect
import tempfile
import cv2
import os

app = FastAPI()

@app.get("/")
def greet():
    return {"message": "Bonjour"}

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    try:
        # Lire le fichier vidéo en tant qu'octets
        video_bytes = await file.read()

        # Créer un fichier temporaire pour stocker les données vidéo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name

        # Vérifier si le fichier vidéo peut être ouvert avec OpenCV
        cap = cv2.VideoCapture(temp_file_path)
        
        if not cap.isOpened():
            raise ValueError("Erreur : Impossible de lire la vidéo")

        # Détection des véhicules et plaques
        detected_cars = car_detect(cap)  # Exécution de car_detect avec le flux vidéo

        # Libérer la ressource vidéo
        cap.release()

        # Supprimer le fichier temporaire
        os.remove(temp_file_path)

        return {"message": "Traitement vidéo réussi", "detected_cars": detected_cars}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Une erreur est survenue : {str(e)}"},
        )
