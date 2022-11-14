import os
import gradio as gr
from modules import scripts, script_callbacks
import modules
from modules.ui import create_refresh_button
from modules import sd_hijack, shared
from modules.paths import script_path
from webui import wrap_gradio_gpu_call
import scripts.dream_artist as dream_artist

def on_ui_train_tabs(params):
    txt2img_preview_params=params.txt2img_preview_params

    with open('./log.txt', 'w', encoding='utf8') as f2:
        f2.write('on_ui_train_tabs')
    with gr.Tab(label="DreamArtist Create embedding"):
        new_embedding_name = gr.Textbox(label="Name")
        initialization_text = gr.Textbox(label="Initialization text", value="*")
        nvpt = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=3, interactive=True)
        use_negative = gr.Checkbox(label='Use negative embedding (DreamArtist)', value=True)
        nvpt_neg = gr.Slider(label="Number of negative vectors per token", minimum=1, maximum=75, step=1, value=6, interactive=True)
        overwrite_old_embedding = gr.Checkbox(value=False, label="Overwrite Old Embedding")

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML(value="")

            with gr.Column():
                create_embedding = gr.Button(value="Create embedding", variant='primary')

    with gr.Tab(label="DreamArtist Train"):
        gr.HTML(
            value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
        with gr.Row():
            train_embedding_name = gr.Dropdown(label='Embedding', elem_id="train_embedding",
                                               choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
            create_refresh_button(train_embedding_name, sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings,
                                  lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())},
                                  "refresh_train_embedding_name")
        with gr.Row():
            train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork",
                                                  choices=[x for x in shared.hypernetworks.keys()])
            create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks,
                                  lambda: {"choices": sorted([x for x in shared.hypernetworks.keys()])}, "refresh_train_hypernetwork_name")
        with gr.Row():
            embedding_learn_rate = gr.Textbox(label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005")
            hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate', placeholder="Hypernetwork Learning rate", value="0.00001")

        # support DreamArtist
        gr.HTML(value='<p style="margin-bottom: 0.7em">DreamArtist</p>')
        with gr.Row():
            neg_train = gr.Checkbox(label='Train with DreamArtist', value=True)
            rec_train = gr.Checkbox(label='Train with reconstruction', value=False)
        cfg_scale = gr.Number(label='CFG scale', value=5.0)
        rec_loss_w = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label="Reconstruction loss weight", value=1.0, interactive=True)
        neg_lr_w = gr.Slider(minimum=0.2, maximum=5.0, step=0.05, label="Negative lr weight", value=1.0, interactive=True)
        disc_path = gr.Textbox(label='Classifier path', placeholder="Path to classifier ckpt, can be empty", value="")

        batch_size = gr.Number(label='Batch size', value=1, precision=0)
        dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images")
        log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value="textual_inversion")
        template_file = gr.Textbox(label='Prompt template file',
                                   value=os.path.join(script_path, "textual_inversion_templates", "style_filewords.txt"))
        training_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512, interactive=True)
        training_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512, interactive=True)
        steps = gr.Number(label='Max steps', value=100000, precision=0)
        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0)
        save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0)
        save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True)
        preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False)

        with gr.Row():
            interrupt_training = gr.Button(value="Interrupt")
            train_embedding = gr.Button(value="Train Embedding", variant='primary')

    def setup():
        create_embedding.click(
            fn=dream_artist.ui.create_embedding,
            inputs=[
                new_embedding_name,
                initialization_text,
                nvpt,
                overwrite_old_embedding,
                use_negative,
                nvpt_neg
            ],
            outputs=[
                train_embedding_name,
                shared.ti_output,
                shared.ti_outcome,
            ]
        )

        train_embedding.click(
            fn=wrap_gradio_gpu_call(dream_artist.ui.train_embedding, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                steps,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params,
                cfg_scale,
                disc_path,
                neg_train,
                rec_train,
                rec_loss_w,
                neg_lr_w
            ],
            outputs=[
                shared.ti_output,
                shared.ti_outcome,
            ]
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )
    params.dream_artist_trigger=setup

    return None


script_callbacks.on_ui_train_tabs(on_ui_train_tabs)