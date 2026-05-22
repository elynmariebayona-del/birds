# 🐦 Bird Detection using YOLOv8

A Streamlit web app that detects birds in images using a fine-tuned YOLOv8n model.

## Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `best.pt` | Trained YOLOv8 model weights *(you must add this manually)* |
| `requirements.txt` | Python dependencies |

## How to Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Add your `best.pt` file to the repo *(download from Colab after training)*
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Click **New app** → connect your GitHub repo
5. Set **Main file path** to `app.py`
6. Click **Deploy**

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
- **Dataset:** Invading Birds (Roboflow)
- **Images:** 899 (< 1,000)
- **Classes:** `bird`
- **Model:** YOLOv8n fine-tuned

## Tech Stack
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io)
- [Pillow](https://python-pillow.org)
