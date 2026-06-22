# Trabajo-Practico-Final-pdi-1c-2026
Repositorio del Trabajo Práctino Final de la materia Técnicas de Procesamiento Digital de Imágenes

# Impacto del Procesamiento de Imágenes sobre Modelos de Hugging Face

## Descripción

Este proyecto analiza cómo distintas técnicas clásicas de procesamiento digital de imágenes modifican las descripciones generadas por un modelo preentrenado de Hugging Face.

La aplicación permite cargar una imagen, aplicar diferentes técnicas de procesamiento y comparar visualmente los resultados junto con la descripción generada automáticamente por el modelo BLIP.

---

## Objetivo

Evaluar el impacto que tienen distintas técnicas de procesamiento de imágenes sobre la interpretación realizada por un modelo de inteligencia artificial para generación de descripciones automáticas.

Las técnicas implementadas son:

* Ecualización del histograma.
* CLAHE (Contrast Limited Adaptive Histogram Equalization).
* Suavizado Gaussiano.
* Ajuste de brillo y contraste.
* Mejora de saturación mediante el espacio de color HSV.

---

## Modelo utilizado

Se utiliza el modelo preentrenado **BLIP (Bootstrapping Language-Image Pre-training)** de Hugging Face para generar descripciones automáticas de las imágenes.

El modelo recibe una imagen y produce una descripción textual en lenguaje natural.

---

## Técnicas de procesamiento implementadas

- **Ecualización del histograma:** mejora el contraste global de la imagen redistribuyendo los niveles de intensidad.

- **CLAHE:** aumenta el contraste de forma adaptativa en regiones pequeñas, evitando la sobreamplificación del ruido.

- **Suavizado Gaussiano:** reduce ruido y detalles finos mediante un filtro de desenfoque.

- **Ajuste de brillo y contraste:** modifica la intensidad luminosa y la diferencia entre zonas claras y oscuras.

- **Mejora en HSV:** realiza una mejora sobre el espacio de color HSV, aumentando la intensidad de los colores y conservando la apariencia natural de la imagen.

---

## Tecnologías utilizadas

* Python
* Gradio
* Hugging Face Transformers
* BLIP
* OpenCV
* NumPy
* Pillow
* PyTorch

---

## Estructura del proyecto

```text
app.py                # Interfaz Gradio
caption_model.py      # Modelo BLIP y generación de captions
preprocessing.py      # Técnicas de procesamiento de imágenes
requirements.txt      # Dependencias del proyecto
README.md             # Documentación
```

---

## Ejecución local

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
python app.py
```

La aplicación se abrirá en una interfaz web local mediante Gradio.

---

## Autor

Trabajo Práctico Final - Procesamiento Digital de Imágenes
Universidad - 2026
