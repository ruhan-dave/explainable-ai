# DeepSeek-Mimic OCR Pipeline

A CPU-optimized explainable AI pipeline for OCR using DeepSeek models, SAM, and LIME.

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend:
   ```bash
   python ocr_backend.py
   ```
   The API will be available at http://localhost:8000

4. Set up frontend (in separate terminal):
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

Switch between Demo Mode (mock data) and Live Mode (connects to backend).
