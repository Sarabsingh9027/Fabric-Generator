# Fabric Roll Detection and Color Variation

This project provides a comprehensive solution for detecting fabric rolls in images, classifying them as single or grouped, extracting their dominant colors, and generating color variations based on a reference fabric. It leverages state-of-the-art computer vision models like YOLOv8 and Segment Anything Model (SAM), along with custom image processing techniques.

The solution is encapsulated within a FastAPI backend for robust API endpoints and a Streamlit frontend for an interactive user interface.

## Features

- **Fabric Roll Detection**: Utilizes a fine-tuned YOLOv8 model to accurately detect and localize fabric rolls in images.
- **Image Classification**: Classifies images as containing a 'single' fabric roll or 'grouped' fabric rolls using a custom-built classifier.
- **Dominant Color Extraction**: Extracts the dominant color from detected fabric rolls using K-Means clustering.
- **Color Variation Generation**: Recolors a single fabric roll image using the dominant colors extracted from a grouped fabric image, allowing for visual exploration of different colorways.
- **FastAPI Backend**: Exposes API endpoints for classification, detection, color extraction, and variation generation.
- **Streamlit Frontend**: Provides an intuitive web interface to interact with the backend services.

## Project Structure

```
├── foto/
│   ├── assets/
│   │   ├── images/
│   │   ├── labels/
│   │   ├── labels_backup/
│   │   └── runs/
│   ├── classification_check.png
│   ├── color_results.png
│   ├── detection_results.csv
│   ├── fabric_variations_v2.png
│   └── label_verification.png
├── groundingdino_repo/ (cloned GroundingDINO repository)
├── weights/
│   └── groundingdino_swint_ogc.pth
├── best.pt (YOLOv8 trained model weights)
├── classifier.py (custom image classifier)
├── main.py (FastAPI application)
├── streamlit_app.py (Streamlit frontend)
├── dataset.yaml (YOLOv8 dataset configuration)
├── requirements.txt
└── README.md
```

## Installation

1.  **Clone the repository (or open in Colab)**:

    ```bash
    git clone https://github.com/your-username/fabric-roll-detection.git
    cd fabric-roll-detection
    ```

2.  **Install dependencies**:

    It's recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    *Note*: Some dependencies are installed directly within the Colab notebook. Ensure `groundingdino-py`, `supervision`, `ultralytics`, `segment-anything`, `torchvision`, `transformers`, `tokenizers`, `scikit-learn`, `opencv-python-headless`, `pillow`, `numpy`, `fastapi`, `uvicorn`, `python-multipart`, `streamlit`, `requests`, `python-dotenv` are correctly installed.

3.  **Download pre-trained weights for Grounding DINO and SAM**:

    These are typically downloaded by the notebook during initial setup.

    -   `groundingdino_swint_ogc.pth`
    -   `sam_vit_b_01ec64.pth`

4.  **Place your YOLOv8 model weights**:

    Ensure your trained YOLOv8 model (`best.pt`) is accessible in the root directory or configure `MODEL_PATH` in `main.py` accordingly. The notebook will automatically save it to `/content/drive/MyDrive/foto/runs/fabric_v2/weights/best.pt`.

## Usage

### 1. Run the FastAPI Backend

Navigate to the project root directory and run the `main.py` script:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

If running in Google Colab, you might need to use `ngrok` or `localtunnel` to expose the port. For example:

```python
!pip install colab_proxy
from colab_proxy import launch_ngrok
launch_ngrok(port=8000)
```

### 2. Run the Streamlit Frontend

Once the FastAPI backend is running, open a new terminal (or Colab cell) and launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

If running in Google Colab, Streamlit will provide a public URL.

## API Endpoints

The FastAPI application exposes the following endpoints:

-   `/health`: Checks the API status.
-   `/classify`: Classifies an uploaded image as 'single' or 'group'.
-   `/detect`: Performs object detection on an uploaded image, returning bounding boxes, confidences, and an annotated image.
-   `/extract-colors`: Extracts dominant colors from fabric rolls in an uploaded image.
-   `/generate-variations`: Takes a single fabric image and a grouped fabric image, and generates color variations of the single fabric using colors from the group image.

## Development Notes

-   **Classifier (`classifier.py`)**: Contains the custom logic to determine if an image has a single large fabric or multiple grouped fabrics.
-   **`_torch_load_with_weights_only_disabled`**: A patch implemented in `main.py` to handle potential `weights_only` issues when loading PyTorch models within certain environments.
-   **Color Recoloring Logic**: The `recolor_fabric` function (in `main.py` and `ovgb9WTL1sad` cell) intelligently recolors fabric, preserving texture and adjusting luminosity.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
