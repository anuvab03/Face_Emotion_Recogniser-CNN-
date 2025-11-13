import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# COLORFUL DARK MODE CSS
# -----------------------
st.markdown(
    """
    <style>
    /* base */
    body, .stApp { background-color: #0b0f14 !important; color: #e6eef6; }

    /* header */
    .header {
        background: linear-gradient(90deg, #7b61ff, #ff6b6b, #ffb86b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 800;
        text-align:center;
        margin-top:6px;
        margin-bottom:0px;
    }
    .subheader { text-align:center; color:#b9c6d9; margin-bottom:20px; font-size:16px; }

    /* card */
    .pred-card {
        background: linear-gradient(180deg, rgba(30,34,41,0.9), rgba(22,26,32,0.8));
        padding: 14px;
        border-radius: 12px;
        margin-top: 12px;
        border: 1px solid rgba(120,120,130,0.12);
        box-shadow: 0 6px 18px rgba(10,10,10,0.6);
        color: #e6eef6;
    }

    /* colorful small badges */
    .badge {
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        font-weight:700;
        color:#071124;
        background: linear-gradient(90deg,#ffd15c,#ff7b7b);
        margin-right:8px;
    }

    /* buttons */
    .stButton>button {
        background: linear-gradient(90deg,#7b61ff,#4fc3f7);
        color: white;
        border-radius: 12px;
        padding: 10px 18px;
        font-size: 16px;
        border:none;
    }
    .stButton>button:hover { filter: brightness(1.08); transform: translateY(-1px); }

    /* progress bar look */
    .conf-bar {
        background: linear-gradient(90deg,#263238,#11151a);
        height:10px;
        border-radius:8px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,0.04);
    }
    .conf-fill { height:100%; border-radius:8px; }

    /* input area */
    .input-box {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        padding:10px;
        border-radius:10px;
        border:1px solid rgba(255,255,255,0.03);
    }

    /* image corners */
    img { border-radius: 12px; }

    /* small text */
    .muted { color:#9fb0c9; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Labels & constants
# -----------------------
IMG_SIZE = (48, 48)
CLASS_MAP = Path("class_indices.npy")
DEFAULT_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOJIS = {0: "😠", 1: "🤢", 2: "😨", 3: "😄", 4: "😭", 5: "😲", 6: "😐"}

# -----------------------
# label loader
# -----------------------
@st.cache_resource
def load_labels():
    if CLASS_MAP.exists():
        try:
            mapping = np.load(CLASS_MAP, allow_pickle=True).item()
            idx_to_label = {v: k for k, v in mapping.items()}
            return [idx_to_label[i] for i in sorted(idx_to_label)]
        except Exception:
            return DEFAULT_CLASSES
    return DEFAULT_CLASSES

# -----------------------
# model loader (ONLY .keras)
# -----------------------
APP_DIR = Path.cwd()
MODELS_DIR = Path("models")
PREF = "emotion_cnn.keras"

def find_keras():
    # preferred names
    p = APP_DIR / PREF
    if p.exists(): return p
    p = MODELS_DIR / PREF
    if p.exists(): return p
    # any .keras in app dir
    anyk = next(APP_DIR.glob("*.keras"), None)
    if anyk: return anyk
    # any .keras in models dir
    if MODELS_DIR.exists():
        anyk = next(MODELS_DIR.glob("*.keras"), None)
        if anyk: return anyk
    return None

# In-page uploader (no sidebar)
uploaded_keras = st.file_uploader("Upload a full Keras model (.keras) — optional", type=["keras"])

uploaded_path = None
if uploaded_keras is not None:
    uploaded_path = APP_DIR / uploaded_keras.name
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_keras.getbuffer())
    st.success(f"Uploaded model: {uploaded_keras.name}")

@st.cache_resource
def load_model_and_labels():
    labels = load_labels()
    # uploaded model has top priority
    if uploaded_path is not None and uploaded_path.exists():
        try:
            m = tf.keras.models.load_model(str(uploaded_path))
            return m, labels
        except Exception as e:
            # cannot show st.* inside cached fn; return None and show later
            return None, labels

    f = find_keras()
    if f is None:
        return None, labels
    try:
        m = tf.keras.models.load_model(str(f))
        return m, labels
    except Exception:
        return None, labels

# -----------------------
# load model
# -----------------------
model, labels = load_model_and_labels()

# show header
st.markdown(f'<div class="header">Facial Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subheader">Capture or upload an image → detect faces → predict emotions</div>', unsafe_allow_html=True)

# top strip with model status and quick controls
colA, colB, colC = st.columns([1, 2, 1])
with colA:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    st.markdown('**Model (auto-detect .keras)**', unsafe_allow_html=True)
    model_file = find_keras()
    if uploaded_path:
        st.markdown(f'<span class="badge">Uploaded</span> {uploaded_path.name}', unsafe_allow_html=True)
    elif model_file:
        st.markdown(f'<span class="badge">Loaded</span> {model_file.name}', unsafe_allow_html=True)
    else:
        st.markdown('<span class="muted">No .keras model found in folder. Upload one above.</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
    st.markdown('<small class="muted">Tip: use well-lit, frontal faces for better predictions</small>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with colC:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    st.markdown('**Classes**', unsafe_allow_html=True)
    lbls = labels if labels is not None else load_labels()
    st.markdown(", ".join(lbls), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Input area: camera + upload (two columns)
# -----------------------
left, right = st.columns([1, 1])
with left:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    st.subheader("📷 Camera")
    camera_img = st.camera_input("Take a photo")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    st.subheader("🖼 Upload Image")
    uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

input_blob = camera_img or uploaded_img

# -----------------------
# helper: preprocess
# -----------------------
def preprocess_roi_gray(roi):
    try:
        face = cv2.resize(roi, IMG_SIZE)
    except Exception:
        return None
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    return face

# detection + annotate
def detect_and_annotate(image_bgr, model, labels):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(36,36))
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = None

    results = []
    for i, (x, y, w, h) in enumerate(faces, start=1):
        pad = max(1, int(0.06 * h))
        y1 = max(0, y + pad); y2 = min(gray.shape[0], y + h - pad)
        x1 = max(0, x + pad); x2 = min(gray.shape[1], x + w - pad)
        roi = gray[y1:y2, x1:x2]
        x_input = preprocess_roi_gray(roi)
        if x_input is None:
            continue
        preds = model.predict(x_input)
        label_idx = int(np.argmax(preds))
        label = labels[label_idx] if label_idx < len(labels) else str(label_idx)
        conf = float(np.max(preds))
        results.append((i, label, conf, (x, y, w, h)))
        # annotate
        draw.rectangle([x, y, x + w, y + h], outline=(120, 255, 180), width=3)
        text = f"{label} {conf:.2f} {EMOJIS.get(label_idx,'')}"
        draw.text((x, max(0, y - 22)), text, fill=(235, 245, 255), font=font)

    return pil, results

# -----------------------
# Run prediction when input given
# -----------------------
if input_blob:
    image = Image.open(input_blob).convert("RGB")
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Input image", width='stretch')

    cols = st.columns([1, 2])
    with cols[0]:
        min_face = st.slider("Min face size (px)", 24, 200, 48)
        analyze = st.button("🔍 Analyze Emotion")
    with cols[1]:
        st.markdown('<div style="padding-top:14px"></div>', unsafe_allow_html=True)

    if analyze:
        if model is None:
            st.error("No .keras model loaded. Upload a full .keras model at the top.")
        else:
            annotated, results = detect_and_annotate(img_np, model, labels)
            if not results:
                st.warning("No faces detected. Try a clearer / closer image.")
            else:
                # show annotated image
                st.image(annotated, caption="Annotated", width='stretch')

                # show per-face cards
                for idx, label, conf, bbox in results:
                    # color based on confidence
                    pct = int(conf * 100)
                    if pct > 75:
                        color = "#7bffb2"
                    elif pct > 50:
                        color = "#ffd86b"
                    else:
                        color = "#ff7b7b"

                    st.markdown(
                        f"""
                        <div class="pred-card">
                            <div style="display:flex;align-items:center;justify-content:space-between;">
                                <div>
                                    <h3 style="margin:0">{EMOJIS.get(labels.index(label) if label in labels else 0,'')} {label}</h3>
                                    <div class="muted">Person {idx}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-weight:700; font-size:18px;">{conf:.2f}</div>
                                    <div style="width:160px; margin-top:8px;" class="conf-bar">
                                        <div class="conf-fill" style="width:{pct}%; background:linear-gradient(90deg,{color}, rgba(255,255,255,0.05));"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

else:
    st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
    st.info("Provide an image from Camera or Upload to begin.")

# footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="muted" style="text-align:center">Model loader looks for a full `.keras` file in the app folder or `models/`. Upload a `.keras` above to use it.</div>', unsafe_allow_html=True)
