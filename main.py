# --- inferencia_gradio_final.py ---

import numpy as np
import cv2
import pickle
import gradio as gr

# --- Parámetros ---
TAMANO_VENTANA = 80
PASO_VENTANA = 10  # Sliding más fino
GRID_SIZE = 4

# --- Funciones Básicas ---


def dividir_en_celdas(imagen, grid_size=GRID_SIZE):
    h, w = imagen.shape
    dh, dw = h // grid_size, w // grid_size
    celdas = []
    for i in range(grid_size):
        for j in range(grid_size):
            celda = imagen[i*dh:(i+1)*dh, j*dw:(j+1)*dw]
            celdas.append(celda)
    return celdas


def proyectar_con_modelo(imagen, modelo):
    celdas = dividir_en_celdas(imagen)
    q_list = []
    for idx, celda in enumerate(celdas):
        vector = celda.flatten()
        W = modelo[idx]['W']
        mean_pca = modelo[idx]['mean_pca']
        q = (vector - mean_pca) @ W
        q_list.append(q)
    return q_list


def log_probabilidad_gaussiana(q, mean, cov):
    k = len(q)
    q_m = q - mean
    try:
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(k))
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(k))
    det_cov = np.linalg.det(cov + 1e-6 * np.eye(k))
    if det_cov <= 0:
        det_cov = 1e-6
    term1 = -0.5 * np.dot(np.dot(q_m, cov_inv), q_m)
    term2 = -0.5 * (k * np.log(2 * np.pi) + np.log(det_cov))
    return term1 + term2


def sliding_window(image, step_size, window_size=(TAMANO_VENTANA, TAMANO_VENTANA)):
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes).astype(float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(
            idxs,
            np.concatenate(
                ([len(idxs) - 1], np.where(overlap > overlap_thresh)[0]))
        )

    return boxes[pick].astype(int)

# --- Detección Principal ---


def evaluar_parche_con_gaussianas(parche, modelo_cara, modelo_nocara):
    q_parche_cara = proyectar_con_modelo(parche, modelo_cara)
    q_parche_nocara = proyectar_con_modelo(parche, modelo_nocara)

    log_prob_cara = 0
    log_prob_nocara = 0

    for idx in range(len(q_parche_cara)):
        qi_cara = q_parche_cara[idx]
        qi_nocara = q_parche_nocara[idx]

        mean_cara = modelo_cara[idx]['mean_gauss']
        cov_cara = modelo_cara[idx]['cov_gauss']
        mean_nocara = modelo_nocara[idx]['mean_gauss']
        cov_nocara = modelo_nocara[idx]['cov_gauss']

        log_prob_cara += log_probabilidad_gaussiana(
            qi_cara, mean_cara, cov_cara)
        log_prob_nocara += log_probabilidad_gaussiana(
            qi_nocara, mean_nocara, cov_nocara)

    score = log_prob_cara - log_prob_nocara
    return score


def detectar_caras_con_gaussianas(imagen_input, lambda_threshold=0.0):
    # Cargar modelos
    with open('./train/modelo_pca_cara_gauss.pkl', 'rb') as f:
        modelo_cara = pickle.load(f)
    with open('./train/modelo_pca_nocara_gauss.pkl', 'rb') as f:
        modelo_nocara = pickle.load(f)

    imagen_original = imagen_input.copy()
    imagen_color = imagen_input.copy()

    imagen_gris_original = cv2.cvtColor(imagen_original, cv2.COLOR_RGB2GRAY)
    imagen_gris_original = imagen_gris_original.astype(np.float32) / 255.0

    escalas = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    detecciones = []

    for escala in escalas:
        nueva_ancho = int(imagen_gris_original.shape[1] * escala)
        nueva_alto = int(imagen_gris_original.shape[0] * escala)

        if nueva_ancho < TAMANO_VENTANA or nueva_alto < TAMANO_VENTANA:
            continue

        imagen_redimensionada = cv2.resize(
            imagen_gris_original, (nueva_ancho, nueva_alto))

        for (x, y, parche) in sliding_window(imagen_redimensionada, PASO_VENTANA):
            if parche.shape[0] != TAMANO_VENTANA or parche.shape[1] != TAMANO_VENTANA:
                continue

            score = evaluar_parche_con_gaussianas(
                parche, modelo_cara, modelo_nocara)

            if score > lambda_threshold:
                x_origen = int(x / escala)
                y_origen = int(y / escala)
                tamano_ventana_original = int(TAMANO_VENTANA / escala)
                detecciones.append(
                    (x_origen, y_origen, x_origen + tamano_ventana_original, y_origen + tamano_ventana_original, score))

    if len(detecciones) > 0:
        cajas = np.array([[x1, y1, x2, y2]
                         for (x1, y1, x2, y2, score) in detecciones])
        detecciones_nms = non_max_suppression(cajas, overlap_thresh=0.3)
    else:
        detecciones_nms = []

    # Asociar cajas y scores para poder pintar texto y color
    detecciones_nms_con_score = []
    for (x1, y1, x2, y2) in detecciones_nms:
        for (dx1, dy1, dx2, dy2, score) in detecciones:
            if abs(x1 - dx1) < 5 and abs(y1 - dy1) < 5 and abs(x2 - dx2) < 5 and abs(y2 - dy2) < 5:
                detecciones_nms_con_score.append((x1, y1, x2, y2, score))
                break

    # Dibujar
    for (x1, y1, x2, y2, score) in detecciones_nms_con_score:
        # Asignar color según score
        if score > -45:
            color = (0, 255, 0)  # Verde
        elif score > -50:
            color = (0, 255, 255)  # Amarillo
        else:
            color = (0, 0, 255)  # Rojo

        cv2.rectangle(imagen_color, (x1, y1), (x2, y2), color, 2)
        texto = f"{score:.1f}"
        cv2.putText(imagen_color, texto, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return imagen_color

# --- Gradio App ---


demo = gr.Interface(
    fn=detectar_caras_con_gaussianas,
    inputs=[
        gr.Image(type="numpy", label="Sube una imagen"),
        gr.Slider(minimum=-70, maximum=70, value=-16, step=1,
                  label="Umbral de Detección (Lambda Threshold)")
    ],
    outputs=gr.Image(type="numpy", label="Imagen con Detecciones"),
    title="Detector de Caras usando PCA + Gaussianas (Schneiderman-Kanade)",
    description="Sube una imagen. Ajusta el umbral para detectar caras basándote en probabilidades gaussianas."
)

if __name__ == "__main__":
    demo.launch()
