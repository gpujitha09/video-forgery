from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt"


def build_model(device: torch.device) -> torch.nn.Module:
    num_classes = 2  # background + forgery
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_model() -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    return model, device


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def sample_frames(video_path: str, n_samples: int = 12) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    indices = np.linspace(0, total_frames - 1, min(n_samples, total_frames), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((int(idx), frame))

    cap.release()
    if not frames:
        raise RuntimeError("Failed to read sampled frames")

    return frames


def score_frame(model: torch.nn.Module, device: torch.device, frame_bgr: np.ndarray) -> float:
    tensor = frame_to_tensor(frame_bgr).to(device)
    with torch.no_grad():
        out = model([tensor])[0]

    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()

    fake_scores = scores[labels == 1] if len(scores) else np.array([])
    return float(fake_scores.max()) if len(fake_scores) else 0.0


def analyze_video(model: torch.nn.Module, device: torch.device, video_path: str, n_samples: int = 12):
    sampled = sample_frames(video_path, n_samples=n_samples)

    frame_scores = []
    for idx, frame in sampled:
        score = score_frame(model, device, frame)
        frame_scores.append({"frame_idx": idx, "score": score, "frame": frame})

    scores = np.array([x["score"] for x in frame_scores], dtype=np.float32)
    max_score = float(scores.max())
    mean_score = float(scores.mean())
    return frame_scores, max_score, mean_score


def main() -> None:
    st.set_page_config(page_title="Video Forgery Detector", layout="wide")
    st.title("Video Forgery Detection Web App")
    st.write("Upload a video and run your trained model to estimate whether it is forged.")

    if not CHECKPOINT_PATH.exists():
        st.error(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return

    threshold = st.slider("Forgery threshold", min_value=0.10, max_value=0.95, value=0.45, step=0.01)
    n_samples = st.slider("Frames sampled from video", min_value=4, max_value=40, value=12, step=2)

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is None:
        st.info("Upload a video file to start analysis.")
        return

    st.video(uploaded)

    if st.button("Analyze Video", type="primary"):
        try:
            model, device = load_model()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                temp_video_path = tmp.name

            with st.spinner("Running inference on sampled frames..."):
                frame_scores, max_score, mean_score = analyze_video(model, device, temp_video_path, n_samples=n_samples)

            pred_label = "FAKE" if max_score >= threshold else "REAL"
            conf = max_score if pred_label == "FAKE" else 1.0 - max_score

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", pred_label)
            c2.metric("Max Forgery Score", f"{max_score:.3f}")
            c3.metric("Confidence", f"{conf:.3f}")

            chart_data = {
                "frame_idx": [x["frame_idx"] for x in frame_scores],
                "score": [x["score"] for x in frame_scores],
            }
            st.line_chart(chart_data, x="frame_idx", y="score")

            st.subheader("Most Suspicious Frames")
            top_k = sorted(frame_scores, key=lambda x: x["score"], reverse=True)[:3]
            cols = st.columns(len(top_k))
            for col, item in zip(cols, top_k):
                frame_rgb = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2RGB)
                col.image(frame_rgb, caption=f"frame={item['frame_idx']} | score={item['score']:.3f}", use_container_width=True)

        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
