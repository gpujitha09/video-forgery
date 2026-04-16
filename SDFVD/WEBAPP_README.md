# Video Forgery Detection Web App

This project is a Streamlit web app that detects likely forged videos using a trained Faster R-CNN model.

## Project Files

- `webapp_streamlit.py`: Streamlit app entry point.
- `requirements-webapp.txt`: Python dependencies for the app.
- `artifacts/checkpoints/best_model.pt`: Trained model checkpoint required at runtime.

## Prerequisites

- Python 3.10+
- pip

## Install Libraries

From the project folder (`SDFVD/SDFVD`):

```powershell
python -m pip install -r requirements-webapp.txt
```

### Required Libraries (from requirements file)

- streamlit>=1.35.0
- torch>=2.3.0
- torchvision>=0.18.0
- opencv-python>=4.9.0
- numpy>=1.26.0

### Installed Libraries (current environment snapshot)

- streamlit==1.56.0
- torch==2.11.0
- torchvision==0.26.0
- opencv-python==4.13.0.92
- numpy==2.4.4

## Run Locally

From the project folder (`SDFVD/SDFVD`):

```powershell
streamlit run webapp_streamlit.py
```

Then open:

- http://localhost:8501

## How To Use

1. Upload a video file (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. Click **Analyze Video**.
3. Review:
   - Prediction: `REAL` or `FAKE`
   - Max forgery score
   - Confidence
   - Frame score chart
   - Most suspicious frames

## Deployment

### Option 1: Streamlit Community Cloud (quickest)

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app.
3. Set:
   - Main file path: `webapp_streamlit.py`
   - Requirements file: `requirements-webapp.txt`
4. Ensure `artifacts/checkpoints/best_model.pt` is included in the repo or downloaded at startup.
5. Deploy.

### Option 2: VM or server (Linux/Windows)

1. Clone the project on the server.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
python -m pip install -r requirements-webapp.txt
```

4. Start the app:

```bash
streamlit run webapp_streamlit.py --server.port 8501 --server.address 0.0.0.0
```

5. Expose port `8501` in firewall/security group.
6. Put Nginx/Caddy reverse proxy in front if using a domain.

## Runtime Notes

- If `best_model.pt` is missing, the app will fail with checkpoint-not-found error.
- CPU inference works, but GPU significantly improves speed.
- Large videos may take longer because frames are sampled and scored individually.
