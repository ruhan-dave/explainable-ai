# DeepSeek-Mimic OCR Pipeline

A CPU-optimized explainable AI pipeline for OCR using DeepSeek models, SAM, and LIME.

## Setup

1. Download the SAM model weights (required before running):
   ```bash
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```
   Place the downloaded `sam_vit_b_01ec64.pth` file in the project root directory.

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend:
   ```bash
   python pipeline.py
   ```
   The API will be available at http://localhost:8000

5. Set up frontend (in separate terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   The frontend will be at http://localhost:5173

## Features

- Vision: CLIP + GradCAM for concept detection
- Verification: Zoom-and-verify with negative prompting
- Structure: MobileSAM for segmentation
- Explainability: Contextual LIME
- Fusion: Chain-of-thought synthesis