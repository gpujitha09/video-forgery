# Video Forgery Detection

Streamlit web application for forged video detection using a trained Faster R-CNN model.

## Repository Structure

- `SDFVD/webapp_streamlit.py` - Streamlit app
- `SDFVD/requirements-webapp.txt` - web app dependencies
- `SDFVD/WEBAPP_README.md` - detailed setup and deployment guide
- `SDFVD/train_video_forgery_detection.ipynb` - training notebook
- `SDFVD/artifacts/checkpoints/best_model.pt` - trained model checkpoint (required at runtime)

## Quick Start

1. Open terminal in the `SDFVD` folder inside this repository.
2. Install dependencies:

```bash
python -m pip install -r requirements-webapp.txt
```

3. Run the app:

```bash
streamlit run webapp_streamlit.py
```

4. Open `http://localhost:8501`.

## Deployment

Use the complete deployment instructions in `SDFVD/WEBAPP_README.md`.

## Notes

- The app requires `artifacts/checkpoints/best_model.pt`.
- If checkpoint is missing, run the training notebook or provide the model file before deployment.
