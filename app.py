# app.py
"""
Streamlit Image Generator + Prompt Enhancer (OpenAI LLM enhancer + providers)

- Fully works with OpenAI for both prompt enhancement (ChatCompletion) and image generation (Images API).
- Replicate and Stability functions are placeholders with clear instructions for wiring specific models.
- Keep your API keys in environment variables or Streamlit Secrets:
    OPENAI_API_KEY
    REPLICATE_API_TOKEN  (optional)
    STABILITY_API_KEY    (optional)

Run:
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."
    streamlit run app.py
"""

import os
import io
import time
import base64
from datetime import datetime

import streamlit as st
from PIL import Image
import requests

# Optional SDK imports; not required for OpenAI path to work
try:
    import openai
except Exception:
    openai = None

# ----------------- Config -----------------
SUPPORTED_SIZES = ["256x256", "512x512", "1024x1024"]
DEFAULT_SIZE = "1024x1024"

# ----------------- Helpers -----------------
def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_display_bytes(img_bytes: bytes):
    """Return an image object that streamlit can display directly"""
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

# ----------------- Prompt Enhancers -----------------
def rule_based_enhancer(short_prompt: str) -> str:
    modifiers = [
        "cinematic composition", "ultra realistic", "dramatic lighting",
        "high detail", "photorealistic", "8k", "shallow depth of field",
        "soft film grain", "award-winning photography style"
    ]
    base = short_prompt.strip()
    to_add = [m for m in modifiers if m.lower() not in base.lower()]
    enhanced = f"{base}, " + ", ".join(to_add) if to_add else base
    return enhanced

def llm_enhancer(short_prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Use OpenAI ChatCompletion to rewrite the prompt. Falls back to rule-based enhancer
    if OPENAI_API_KEY is missing or if the openai package isn't available.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key or openai is None:
        # Fallback silently to rule-based (UI warns if user asked for LLM but key missing)
        return rule_based_enhancer(short_prompt)

    openai.api_key = key
    system_msg = (
        "You are an assistant that rewrites short prompts into image-generation-ready prompts. "
        "Add style, mood, lighting, camera hints (aperture, lens), and keep output concise."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Rewrite this prompt for an image generator: {short_prompt}"}
            ],
            max_tokens=140,
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        # Raise so caller can show the error and fallback if desired
        raise RuntimeError(f"OpenAI LLM enhancer failed: {e}")

# ----------------- Provider Implementations -----------------
def generate_with_openai(prompt: str, size: str = DEFAULT_SIZE) -> bytes:
    """
    Uses OpenAI Images API. Requires OPENAI_API_KEY and openai SDK available.
    Returns image bytes (PNG).
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Please set it in environment or Streamlit secrets.")
    if openai is None:
        raise RuntimeError("openai package not installed. Add 'openai' to requirements.")

    openai.api_key = key
    try:
        # Many openai SDK versions use Image.create returning .data[0].b64_json
        result = openai.Image.create(prompt=prompt, n=1, size=size)
        # Parse result safely
        if isinstance(result, dict):
            data0 = result.get("data", [{}])[0]
            b64 = data0.get("b64_json") or data0.get("b64")
        else:
            data0 = result.data[0]
            b64 = getattr(data0, "b64_json", None) or getattr(data0, "b64", None)
        if not b64:
            raise RuntimeError(f"No base64 image data returned by OpenAI. Response: {result}")
        img_bytes = base64.b64decode(b64)
        return img_bytes
    except Exception as e:
        raise RuntimeError(f"OpenAI image generation failed: {e}")

def generate_with_replicate(prompt: str) -> bytes:
    """
    Placeholder: Replace with replicate client code for a specific model.
    Example approach (pseudo):
        - Create prediction via POST to https://api.replicate.com/v1/predictions
        - Poll for result, extract output URL(s), fetch first image, return bytes.
    This stub raises a helpful error if not configured.
    """
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set. To use Replicate set this environment variable.")
    raise RuntimeError(
        "Replicate provider not wired in this example. Update generate_with_replicate() "
        "with the model slug and REST/SDK calls per Replicate docs."
    )

def generate_with_stability(prompt: str, size: str = DEFAULT_SIZE) -> bytes:
    """
    Placeholder: Replace with Stability API or stability-sdk call.
    Example approach (pseudo):
        - POST to Stability endpoint with prompt / width / height / cfg_scale / steps
        - Parse response (often returns base64 or artifact URLs)
    This stub raises a helpful error if not configured.
    """
    key = os.getenv("STABILITY_API_KEY")
    if not key:
        raise RuntimeError("STABILITY_API_KEY not set. To use Stability set this environment variable.")
    raise RuntimeError(
        "Stability provider not wired in this example. Update generate_with_stability() "
        "with the correct API call per Stability docs."
    )

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Image + Prompt Enhancer", layout="wide")
st.title("🖼️ Image Generator with Prompt Enhancer")

# Sidebar settings
with st.sidebar:
    st.header("Settings & Providers")
    provider = st.selectbox("API Provider", options=["openai", "replicate", "stability"], index=0)
    size = st.selectbox("Image size", options=SUPPORTED_SIZES, index=2)
    use_llm = st.checkbox("Enhance prompt with OpenAI LLM (ChatGPT)", value=True,
                         help="Uses OPENAI_API_KEY to rewrite your short prompt into a detailed prompt.")
    max_history = st.slider("History entries (session)", min_value=1, max_value=50, value=12)
    st.markdown("---")
    st.markdown("Environment variables (do NOT commit keys):")
    st.caption("OPENAI_API_KEY, REPLICATE_API_TOKEN (optional), STABILITY_API_KEY (optional)")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Main layout
col_main, col_history = st.columns([2.2, 1])

with col_main:
    st.subheader("Enter a short prompt")
    prompt = st.text_area("Prompt", value="A cozy cabin in a snowy forest", height=140)
    st.write("Enter a short prompt — the enhancer will expand it for image generation.")
    generate_clicked = st.button("Enhance & Generate")

    if generate_clicked:
        if not prompt.strip():
            st.warning("Please enter a prompt before generating.")
        else:
            # Enhance prompt
            enhanced = None
            if use_llm:
                try:
                    enhanced = llm_enhancer(prompt)
                except Exception as e:
                    st.warning(f"LLM enhancer failed — using rule-based enhancer. ({e})")
                    enhanced = rule_based_enhancer(prompt)
            else:
                enhanced = rule_based_enhancer(prompt)

            st.info(f"Enhanced prompt: **{enhanced}**")

            # Generate using selected provider
            try:
                with st.spinner(f"Generating image using {provider}..."):
                    if provider == "openai":
                        img_bytes = generate_with_openai(enhanced, size=size)
                    elif provider == "replicate":
                        img_bytes = generate_with_replicate(enhanced)
                    elif provider == "stability":
                        img_bytes = generate_with_stability(enhanced, size=size)
                    else:
                        raise RuntimeError("Unknown provider selected.")
                # Display
                st.success("Image generated!")
                st.image(img_bytes, use_column_width=True)

                # Save to history
                entry = {
                    "ts": now_ts(),
                    "provider": provider,
                    "prompt": prompt,
                    "enhanced": enhanced,
                    "image_bytes": img_bytes,
                }
                st.session_state.history.insert(0, entry)
                st.session_state.history = st.session_state.history[:max_history]

                # Download button
                fname = f"image_{int(time.time())}.png"
                st.download_button("Download image", data=img_bytes, file_name=fname, mime="image/png")

            except Exception as e:
                st.error(f"Generation error: {e}")

with col_history:
    st.subheader("History (session)")
    if not st.session_state.history:
        st.info("No generated images yet.")
    else:
        for i, h in enumerate(st.session_state.history):
            with st.expander(f"{h['ts']} — {h['provider']} — {h['prompt'][:60]}"):
                st.markdown(f"**Original:** {h['prompt']}")
                st.markdown(f"**Enhanced:** {h['enhanced']}")
                st.image(h["image_bytes"])
                dlname = f"history_{i}.png"
                st.download_button("Download", data=h["image_bytes"], file_name=dlname, mime="image/png")

# Footer
st.markdown("---")
st.caption("Built with Streamlit — OpenAI path active. Replace Replicate/Stability placeholders with model-specific logic as needed.")
