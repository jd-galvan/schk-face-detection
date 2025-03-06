# 🖼️ schk-face-detection

🚀 **Detección de Rostros basada en Schneiderman & Kanade** usando OpenCV y Gradio.

Este proyecto implementa un sistema de detección de rostros en imágenes utilizando un clasificador basado en Haar Cascades, inspirado en el trabajo de Schneiderman y Kanade. Se proporciona una interfaz web interactiva con **Gradio** para subir imágenes y visualizar los resultados.

## 📦 Instalación

Antes de ejecutar la aplicación, asegúrate de tener **Python 3.8 o superior** instalado. Luego, sigue estos pasos:

### 🔹 Crear y activar un entorno virtual (opcional pero recomendado)

Ejecuta los siguientes comandos en la terminal:

```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows
venv\Scripts\activate

# En macOS y Linux
source venv/bin/activate
```

### 🔹 Instalar dependencias

```bash
pip install opencv-python-headless numpy gradio
```

## 🚀 Cómo ejecutar la aplicación

Para iniciar la aplicación, simplemente ejecuta el siguiente comando en la terminal desde la raíz del proyecto:

```bash
python main.py
```

Esto abrirá un enlace en tu navegador donde podrás subir imágenes y ver los rostros detectados.

## 📂 Estructura del proyecto

```
📂 schk-face-detection
├── main.py       # Código principal con la detección de rostros
├── README.md     # Este archivo con la documentación
└── requirements.txt  # Lista de dependencias (opcional)
```

## 🖼️ Uso

1. Sube una imagen a la interfaz de Gradio.
2. El sistema detectará los rostros en la imagen.
3. Se mostrarán los resultados con los rostros resaltados.

## 🛠️ Tecnologías usadas

- **Python** 🐍
- **OpenCV** 👀
- **NumPy** 🔢
- **Gradio** 🎛️

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto, siéntete libre de hacer un *fork* y enviar un *pull request*.