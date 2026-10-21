import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        "triton==3.3.1",
        "xformers==0.0.31",
        "numpy==2.2.6",
        "scipy==1.16.0",
        "tokenizers==0.21.1",
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "huggingface_hub==0.34.2",
        "hf-transfer==0.1.9",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "bitsandbytes==0.46.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("capstone-paper-lock-notebook")


@app.function(image=image)
def notebook_image():
    """Expose the paper-locked image for selection in Modal Notebooks."""
