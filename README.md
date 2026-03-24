# Vehicle Detection System

YOLOv11m · ByteSORT / BoT-SORT · HSV Colour Detection · Streamlit UI

## Setup

```bash
pip install -e ".[dev]"
```

## Run

```bash
PYTHONPATH=src streamlit run src/vehicle_detection/app.py
```

## Test

```bash
pytest
```

## Project Structure

```
Vehicle_Detection/
├── src/
│   └── vehicle_detection/
│       ├── __init__.py
│       ├── app.py          # Streamlit UI (Batch + Real-time tabs)
│       └── utils/
│           ├── color_detection.py
│           └── video_processor.py
├── models/
│   └── best.pt             # Trained YOLOv11m weights (9 classes)
├── tests/
│   └── test_utils.py
├── temp/                   # Runtime temp files (git-ignored)
├── .env                    # Secrets (git-ignored)
├── .gitignore
└── pyproject.toml
```
