import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from core.detection import process_frame
from core.decision import evaluate_risk

st.markdown("""
    <style>
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        max-height: 350px !important;
        overflow-y: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="VISION 2 DECISION", layout="wide")

st.markdown("""
    <h1 style='margin-top:-50px;'>
    VISION 2 DECISION : AI-based Risk Detection System using YOLOv8 🚦<br>
    <span style='font-size:20px;'>
    Experience the Brain of a Self-driving Car 🚗
    </span>
    </h1>
""", unsafe_allow_html=True)
# st.title('''VISION 2 DECISION : AI-based Risk Detection System using YOLOv8 🚦
#           Experience the Brain of a Self-driving Car 🚗 ''')

# ---------------- SIDEBAR ----------------
st.sidebar.header("Control Panel 🎮")



CONF_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05
)

uploaded_file = st.file_uploader(
    "Upload a demo video to see the model in action! 🎥",
    type=["mp4", "avi", "mov"]
)

# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class_names = list(model.names.values())

selected_classes = st.sidebar.multiselect(
    "Select Classes to Detect",
    class_names,
    default=class_names
)

st.sidebar.write(f"Active Classes: {len(selected_classes)}")

# ---------------- VIDEO SELECTION ----------------
sample_video_path = "data/sample.mp4"

if uploaded_file is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())
    video_path = temp_video.name
    st.sidebar.success("Using uploaded video 🎥")
else:
    video_path = sample_video_path
    st.sidebar.info("Using sample video 📁")

# ---------------- VIDEO PROCESSING ----------------
cap = cv2.VideoCapture(video_path)

stframe = st.empty()

frame_count = 0
last_processed = None   # for smooth playback

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO every 3rd frame
    if frame_count % 3 == 0:
        processed_frame = process_frame(
            model,
            frame,
            evaluate_risk,
            CONF_THRESHOLD,
            selected_classes
        )
        last_processed = processed_frame
    else:
        # Show last processed frame for smooth video
        if last_processed is not None:
            processed_frame = last_processed
        else:
            processed_frame = frame

    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    stframe.image(processed_frame, channels="RGB", width=800)

cap.release()

st.success("Video processing completed!")







