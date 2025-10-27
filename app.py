import streamlit as st
import requests
from groq import Groq
import io
from PIL import Image
import base64

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="AI Image Generator with Prompt Enhancer", layout="wide")

st.title("🎨 AI Image Generator with Prompt Enhancer")
st.write("Type a short prompt — the AI will enhance it and generate an image using your selected provider.")

# ------------------------------
# API KEYS FROM STREAMLIT SECRETS
# ------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")
STABILITY_API_KEY = st.secrets.get("STABILITY_API_KEY")

# ------------------------------
# PROMPT ENHANCEMENT (GROQ)
# ------------------------------
def enhance_prompt_groq(user_prompt):
    """Enhance user prompt using Groq LLaMA 3.1 model"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a creative prompt enhancer for AI image generation. Make prompts vivid, cinematic, detailed, and realistic."},
                {"role": "user", "content": user_prompt}
            ]
        )
        enhanced_prompt = response.choices[0].message.content.strip()
        return enhanced_prompt
    except Exception as e:
        st.error(f"Groq prompt enhancement failed: {e}")
        return user_prompt

# ------------------------------
# HUGGING FACE IMAGE GENERATION (updated)
# ------------------------------
def generate_image_huggingface(prompt):
    """Generate image using Hugging Face Stable Diffusion v1-5"""
    try:
        api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {"inputs": prompt}

        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            image_bytes = response.content
            return Image.open(io.BytesIO(image_bytes))
        else:
            st.error(f"Hugging Face API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Hugging Face generation failed: {e}")
        return None


# ------------------------------
# STABILITY AI IMAGE GENERATION (updated)
# ------------------------------
def generate_image_stability(prompt):
    """Generate image using Stability AI new model name"""
    try:
        api_url = "https://api.stability.ai/v2beta/stable-image/generate/core"
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "application/json"
        }
        payload = {
            "prompt": prompt,
            "output_format": "png",
        }

        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code != 200:
            st.error(f"Stability AI Error: {response.status_code} - {response.text}")
            return None

        data = response.json()
        image_base64 = data.get("image")
        if image_base64:
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes))
        else:
            st.error("No image data returned from Stability AI.")
            return None
    except Exception as e:
        st.error(f"Stability AI generation failed: {e}")
        return None

# ------------------------------
# APP UI
# ------------------------------
user_prompt = st.text_area("📝 Enter your prompt:", height=100)
model_choice = st.selectbox(
    "🧠 Select Image Generation Provider",
    ["Hugging Face (Stable Diffusion)", "Stability AI"]
)

if st.button("✨ Generate Image"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Enhancing prompt using Groq..."):
            enhanced = enhance_prompt_groq(user_prompt)
            st.write("**Enhanced Prompt:**", enhanced)

        with st.spinner(f"Generating image via {model_choice}..."):
            if model_choice == "Hugging Face (Stable Diffusion)":
                img = generate_image_huggingface(enhanced)
            else:
                img = generate_image_stability(enhanced)

        if img:
            st.image(img, caption="🖼️ Generated Image", use_container_width=True)

            # Download Button
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="⬇️ Download Image",
                data=byte_im,
                file_name="generated_image.png",
                mime="image/png"
            )
