import cv2
import gradio as gr
import numpy as np


def detect_faces(image):
    # Cargar clasificador de Haar (modelo inspirado en Schneiderman & Kanade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convertir imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image


# Crear la interfaz en Gradio
iface = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Detección de Rostros (Schneiderman & Kanade)",
    description="Sube una imagen y el sistema detectará los rostros usando un algoritmo basado en Schneiderman y Kanade."
)

# Ejecutar la aplicación
if __name__ == "__main__":
    iface.launch()
