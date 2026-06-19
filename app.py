import gradio as gr

from caption_model import generate_caption
from preprocessing import (
    equalization,
    gaussian_blur,
    apply_clahe
)

def process(image):

    original_caption = generate_caption(image)

    equalized = equalization(image)
    equalized_caption = generate_caption(equalized)

    blurred = gaussian_blur(image)
    blurred_caption = generate_caption(blurred)

    clahe = apply_clahe(image)
    clahe_caption = generate_caption(clahe)

    return (
        equalized,
        blurred,
        clahe,
        original_caption,
        equalized_caption,
        blurred_caption,
        clahe_caption
    )

interface = gr.Interface(
    fn=process,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Ecualización"),
        gr.Image(label="Suavizado Gaussiano"),
        gr.Image(label="CLAHE"),
        gr.Textbox(label="Descripción original"),
        gr.Textbox(label="Descripción ecualizada"),
        gr.Textbox(label="Descripción suavizada"),
        gr.Textbox(label="Descripción CLAHE")
    ],
    title="Sistema Inteligente de Descripción de Imágenes",
    description="Comparación del impacto del procesamiento digital sobre modelos de descripción automática."
)

interface.launch()