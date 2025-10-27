import streamlit as st
import requests
from groq import Groq
import io
from PIL import Image

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="AI Image Generator with Prompt Enhancer", layout="wide")

st.title("🎨 AI Image Generator with Prompt Enhancer")
st.write("Type a short prompt — I'll enhance it using AI and generate an image using your selected model.")

# ------------------------------
# Load API Keys Securely
# ------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")
STABILITY_API_KEY = st.secrets.get("STABILITY_API_KEY")

# ------------------------------
# Helper Functions
# ------------------------------
def enhance_prompt_groq(user_prompt):
    """Enhance user prompt using Groq (text model)"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # good general-purpose model
            messages=[
                {"role": "system", "content": "You are an AI prompt enhancer for image generation. Add artistic, realistic, and cinematic style details."},
                {"role": "user", "content": user_prompt}
            ]
        )
        enhanced_prompt = response.choices[0].message.content.strip()
        return enhanced_prompt
    except Exception as e:
        st.error(f"Groq prompt enhancement failed: {e}")
        return user_prompt


def generate_image_huggingface(prompt):
    """Generate image from Hugging Face (Stable Diffusion)"""
    try:
        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {"inputs": prompt}

        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            image_bytes = response.content
            return Image.open(io.BytesIO(image_bytes))
        else:
            st.error(f"Hugging Face API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Hugging Face generation failed: {e}")
        return None


def generate_image_stability(prompt):
    """Generate image using Stability AI API"""
    try:
        api_url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "application/json"
        }
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "height": 512,
            "width": 512,
            "samples": 1
        }

        response = requests.post(api_url, headers=headers, json=payload)
        data = response.json()
        if "artifacts" in data and len(data["artifacts"]) > 0:
            import base64
            image_base64 = data["artifacts"][0]["base64"]
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes))
        else:
            st.error("Stability AI returned no image data.")
            return None
    except Exception as e:
        st.error(f"Stability AI generation failed: {e}")
        return None


# ------------------------------
# UI Components
# ------------------------------
user_prompt = st.text_area("📝 Enter your prompt:", height=100)
model_choice = st.selectbox(
    "🧠 Select Image Generation API",
    ["Hugging Face (Stable Diffusion)", "Stability AI"]
)

if st.button("✨ Generate Image"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Enhancing your prompt using Groq..."):
            enhanced = enhance_prompt_groq(user_prompt)
            st.write("**Enhanced Prompt:**", enhanced)

        with st.spinner(f"Generating image via {model_choice}..."):
            if model_choice == "Hugging Face (Stable Diffusion)":
                img = generate_image_huggingface(enhanced)
            else:
                img = generate_image_stability(enhanced)

        if img:
            st.image(img, caption="Generated Image", use_container_width=True)

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

# ------------------------------
# Optional: Generation History
# ------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("📜 Save to History"):
    if user_prompt:
        st.session_state["history"].append({"prompt": user_prompt})
        st.success("Saved prompt to history!")

if st.session_state["history"]:
    st.subheader("🕓 Generation History")
    for item in st.session_state["history"]:
        st.write("•", item["prompt"])
