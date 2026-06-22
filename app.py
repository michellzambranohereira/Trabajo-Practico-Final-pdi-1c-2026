import gradio as gr

from caption_model import generate_caption

from preprocessing import (
    equalization,
    gaussian_blur,
    apply_clahe,
    adjust_brightness_contrast,
    hsv_h,
    hsv_s,
    hsv_v
)


def process(image, filters):

    gallery = []
    descriptions = ""

    # Original
    gallery.append((image, "Original"))

    original_caption = generate_caption(image)

    descriptions += (
        "ORIGINAL\n"
        f"{original_caption}\n\n"
        "----------------------------------\n\n"
    )

    # Ecualización
    if "Ecualización" in filters:

        img = equalization(image)

        gallery.append((img, "Ecualización"))

        descriptions += (
            "ECUALIZACIÓN\n"
            f"{generate_caption(img)}\n\n"
            "----------------------------------\n\n"
        )

    # CLAHE
    if "CLAHE" in filters:

        img = apply_clahe(image)

        gallery.append((img, "CLAHE"))

        descriptions += (
            "CLAHE\n"
            f"{generate_caption(img)}\n\n"
            "----------------------------------\n\n"
        )

    # Suavizado
    if "Suavizado Gaussiano" in filters:

        img = gaussian_blur(image)

        gallery.append((img, "Suavizado Gaussiano"))

        descriptions += (
            "SUAVIZADO GAUSSIANO\n"
            f"{generate_caption(img)}\n\n"
            "----------------------------------\n\n"
        )

    # Brillo y contraste
    if "Brillo y Contraste" in filters:

        img = adjust_brightness_contrast(image)

        gallery.append((img, "Brillo y Contraste"))

        descriptions += (
            "BRILLO Y CONTRASTE\n"
            f"{generate_caption(img)}\n\n"
            "----------------------------------\n\n"
        )

    # HSV H
    if "HSV (Canal H)" in filters:

        img = hsv_h(image)

        gallery.append((img, "HSV - H"))

        descriptions += (
            "HSV (CANAL H)\n"
            f"{generate_caption(img)}\n\n"
            "----------------------------------\n\n"
        )

    # HSV S
    if "HSV (Canal S)" in filters:

        img = hsv_s(image)

        gallery.append((img, "HSV - S"))

        descriptions += (
            "HSV (CANAL S)\n"
            f"{generate_caption(img)}\n\n"
            "----------------------------------\n\n"
        )

    # HSV V
    if "HSV (Canal V)" in filters:

        img = hsv_v(image)

        gallery.append((img, "HSV - V"))

        descriptions += (
            "HSV (CANAL V)\n"
            f"{generate_caption(img)}\n"
        )

    return gallery, descriptions



with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Impacto del Procesamiento de Imágenes sobre Modelos de Hugging Face

        Esta aplicación compara cómo distintas técnicas clásicas
        de procesamiento modifican las descripciones generadas por
        un modelo BLIP preentrenado de Hugging Face.
        """
    )

    image_input = gr.Image(
        type="pil",
        label="Suba una imagen"
    )

    filters = gr.CheckboxGroup(

        choices=[

            "Ecualización",

            "CLAHE",

            "Suavizado Gaussiano",

            "Brillo y Contraste",

            "HSV (Canal H)",

            "HSV (Canal S)",

            "HSV (Canal V)"
        ],

        label="Seleccione las técnicas a comparar"
    )

    with gr.Row():

        btn = gr.Button("🔍 Comparar")

        clear_btn = gr.Button("🗑️ Limpiar")


    gallery = gr.Gallery(
        label="Imágenes comparadas",
        columns=3,
        height=450
    )

    descriptions = gr.Textbox(
        label="Descripciones generadas por BLIP",
        lines=20
    )


    btn.click(
        fn=process,
        inputs=[image_input, filters],
        outputs=[gallery, descriptions]
    )


    clear_btn.click(
        fn=lambda: (
            None,      # imagen
            [],        # filtros
            None,      # galeria
            ""         # descripciones
        ),

        inputs=[],

        outputs=[
            image_input,
            filters,
            gallery,
            descriptions
        ]
    )


demo.launch(share=True)