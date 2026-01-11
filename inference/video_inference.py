import streamlit as st
import cv2
import torch
import time
from pathlib import Path
import sys
from PIL import Image
import numpy as np

# Add project path
sys.path.append(str(Path(__file__).parent))

# Import your inference code
from infer_smallcnn_video import build_model, build_preprocess, load_class_names


# ---------------------------
# Video Inference Wrapper
# ---------------------------
class VideoInferenceWrapper:
    def __init__(self, model_type="squeezenet", weights_path=None, classes_path="inference/classes.txt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        self.classes = load_class_names(classes_path) if classes_path else None
        self.num_classes = len(self.classes) if self.classes else 6

        self.model = build_model(self.model_type, self.num_classes, self.device)

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state)

        self.model.eval()
        self.preprocess = build_preprocess(self.model_type)

    def predict(self, frame: np.ndarray):
        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            pred_index = output.argmax(dim=1).item()
        if self.classes:
            return self.classes[pred_index]
        return pred_index


# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(
    page_title="EcoBot - Live Plant Health",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("EcoBot - Smart Greenhouse Monitoring")
st.markdown("---")

# ---------------------------
# Load model once
# ---------------------------
if "model" not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model = VideoInferenceWrapper(
            model_type="squeezenet",
            weights_path="best_squeezenet.pt",
            classes_path="inference/classes.txt"
        )

model = st.session_state.model

# ---------------------------
# Load video once
# ---------------------------
VIDEO_PATH = "video.mp4"
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(VIDEO_PATH)

cap = st.session_state.cap

# ---------------------------
# Layout: Video on left, chat on right
# ---------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Live Video Analysis")
    frame_area = st.empty()  # placeholder for video frame

with right_col:
    st.subheader("Chat Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div style="background:#e3f2fd;padding:10px;border-radius:8px;margin:5px;"><b>You:</b> {message["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:#f1f8e9;padding:10px;border-radius:8px;margin:5px;"><b>EcoBot:</b> {message["content"]}</div>',
                    unsafe_allow_html=True)

# ---------------------------
# Process one frame per rerun
# ---------------------------
ret, frame = cap.read()
if not ret:
    # Loop video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()

if ret:
    prediction = model.predict(frame)

    # Draw prediction on frame
    cv2.putText(frame, str(prediction), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert BGR -> RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_area.image(frame_rgb, channels="RGB")

    # Update chat messages if prediction changed
    if "last_prediction" not in st.session_state or st.session_state.last_prediction != prediction:
        st.session_state.last_prediction = prediction
        suggested_question = f"Observation: {prediction}. What does this mean for my plants?"
        st.session_state.messages.append({"role": "user", "content": suggested_question})
        # Dummy bot response (replace with EcobotInterface query)
        bot_response = f"EcoBot advice for: {prediction}"
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# ---------------------------
# Force rerun for continuous video
# ---------------------------
time.sleep(0.03)  # optional: control speed
# Flag to trigger rerun
if "rerun" not in st.session_state:
    st.session_state.rerun = False

# At the end of frame processing
st.session_state.rerun = not st.session_state.rerun

