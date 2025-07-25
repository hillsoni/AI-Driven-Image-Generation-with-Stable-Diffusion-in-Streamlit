import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionPipeline
from transformers import pipeline as hf_pipeline, set_seed
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------------
# Config Class (CFG)
# ----------------------
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_steps = 20  # Reduced for speed
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12
    use_auth_token = os.getenv('HF_TOKEN')  # Loaded securely

# ----------------------
# Set random seed
# ----------------------
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(CFG.seed)

# ----------------------
# Load Stable Diffusion Pipeline
# ----------------------
@st.cache_resource(show_spinner=False)
def load_sd_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id,
        torch_dtype=torch.float16 if CFG.device == 'cuda' else torch.float32,
        use_auth_token=CFG.use_auth_token
    )
    return pipe.to(CFG.device)

# ----------------------
# Load GPT-2 Pipeline
# ----------------------
@st.cache_resource(show_spinner=False)
def load_gpt2_pipeline():
    return hf_pipeline(
        "text-generation",
        model=CFG.prompt_gen_model_id,
        device=0 if CFG.device == 'cuda' else -1
    )

# ----------------------
# Image Generation
# ----------------------
def generate_image(prompt):
    pipe = load_sd_pipeline()
    generator = torch.Generator(CFG.device).manual_seed(CFG.seed)

    if CFG.device == 'cuda':
        with torch.autocast('cuda'):
            image = pipe(prompt, num_inference_steps=CFG.image_gen_steps,
                         guidance_scale=CFG.image_gen_guidance_scale,
                         generator=generator).images[0]
    else:
        image = pipe(prompt, num_inference_steps=CFG.image_gen_steps,
                     guidance_scale=CFG.image_gen_guidance_scale,
                     generator=generator).images[0]

    img_np = np.array(image)
    img_resized = cv2.resize(img_np, CFG.image_gen_size, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(img_resized)

# ----------------------
# GPT-2 Prompt Suggestions
# ----------------------
def generate_prompts(seed_prompt, n_prompts=CFG.prompt_dataset_size, max_length=CFG.prompt_max_length):
    pipe = load_gpt2_pipeline()
    set_seed(CFG.seed)
    outputs = pipe(seed_prompt, max_length=max_length, num_return_sequences=n_prompts,
                   do_sample=True, pad_token_id=50256)
    return [o['generated_text'] for o in outputs]

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="AI Image Generator", layout="centered")
st.title("🖼️ AI Image Generator with Stable Diffusion")

st.markdown("Generate stunning images using Stable Diffusion v2 and optional GPT-2 prompt inspiration.")

prompt = st.text_input("Enter your prompt", value="a futuristic city at sunset")

with st.expander("✨ Need ideas? Generate prompts with GPT-2"):
    seed_prompt = st.text_input("Enter a seed for GPT-2 prompt", value="a ")
    if st.button("Suggest Prompts"):
        with st.spinner("Generating suggestions..."):
            suggestions = generate_prompts(seed_prompt)
        st.write("**Prompt Suggestions:**")
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"{i}. {suggestion}")

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            img = generate_image(prompt)
        st.image(img, caption="Generated Image", use_column_width=True)
        st.success("Image generated!")

st.markdown("""
---
### 🧪 How to Run This App Locally
1. Create `.env` file with your Hugging Face token: `HF_TOKEN=your_token_here`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
""")
