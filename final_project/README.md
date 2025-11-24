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

4. Run the application:
   ```bash
   python pipeline.py
   ```
   Open http://localhost:8000 in your browser to access the web interface.

## Docker Deployment

For production deployment or easier setup:

1. Build the Docker image:
   ```bash
   docker build -t deepseek-mimic .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 deepseek-mimic
   ```

3. Access at http://localhost:8000

The Dockerfile automatically downloads SAM weights during build.

## Features

- Vision: CLIP + GradCAM for concept detection
- Verification: Zoom-and-verify with negative prompting
- Structure: MobileSAM for segmentation
- Explainability: Contextual LIME
- Fusion: Chain-of-thought synthesis

## Pipeline Details
This project is built around a FastAPI (Python) backend that orchestrates a vision-and-explainability pipeline using CLIP from OpenAI, SAM from Meta, and classic XAI tools like GradCAM++ and LIME. The goal is not to only detect and label elements in an image, but to also show where the models are looking and why they made a particular decision.

CLIP is responsible for zero-shot classification. For generating visual attention maps, we use a GradCAM++ implementation. SAM handles segmentation: it finds and refines object masks in the image. GradCAM++ is used to visualize attention in CLIP’s vision backbone, and LIME gives us a local, feature-level explanation for individual predictions.

The system runs through a 6-step pipeline:

1. Vision (Global Attention)
The pipeline starts by asking: where does the model look when it searches for a given concept (e.g., for the DeepSeek OCR diagram, “SAM”, “ViT”)?

We run GradCAM++ on CLIP, hooking into the final visual layers (vision_model.encoder.layers...). For each user-provided text prompt, the system computes a heatmap showing pixel-level activation intensity. This gives a global view of which regions of the image CLIP finds relevant to each concept, before we even talk about specific objects.

2. Segmentation (Object Discovery)
Next, the system tries to discover all the distinct “things” in the image, without yet knowing what they are.

Using SAM’s AutomaticMaskGenerator, we scan the entire image and produce binary masks representing candidate objects or regions. To keep the results meaningful, we filter out masks that are too small (noise) or too large (e.g., broad background regions). For efficiency, crop layers are explicitly disabled (crop_n_layers=0), which speeds up processing without sacrificing the quality we need for later steps.

3. Emergence (Classification)
Once we have a collection of anonymous segments, we need to give them semantic meaning.

The system iterates over each mask from Step 2 and crops the original image to that mask’s bounding box. Each crop is fed into CLIP alongside a set of candidate labels: both user-specified prompts and common diagram elements such as “Arrow”, “Box”, or “Diagram”. CLIP performs zero-shot classification on each crop. If the top label’s confidence exceeds a threshold (0.25), that segment is considered “recognized” and is assigned that label.

4. Fusion (Visualization)
To make the results understandable at a glance, the system overlays them onto the original image.

For every recognized segment, we draw bounding boxes using OpenCV (cv2.rectangle) and render the predicted labels with cv2.putText. The output is a single, fused result image that shows all detected elements and their labels in context. This is what the user sees as the primary visual summary of the model’s understanding.

5. Structure Verification (Refinement)
Often, a user wants to drill down into a single element—for example, clicking on a label like “ViT” to understand it more deeply.

When a user selects a target, the system searches the classified segments from Step 3 for the best textual match. It then takes that segment’s bounding box and feeds it back into SAM, this time through the interactive SAM Predictor rather than the automatic generator. This produces a high-quality, precise segmentation mask focused on just that one object, giving us a clean region of interest for deeper analysis.

6. LIME Analysis (Deep Explanation)
Finally, the system explains why CLIP recognized that specific object as it did—for instance, why a particular region is considered an “arrow.”

First, we extract the region of interest using the refined mask from Step 5. Inside this ROI, we run SLIC superpixel segmentation (skimage.segmentation.slic) to break the object into smaller, coherent subregions. 

LIME then perturbs these superpixels by randomly turning them on and off (e.g., graying some out) and generating many slightly modified versions of the ROI. For each perturbed image, we record CLIP’s confidence for the chosen label.

By fitting a linear regression model to the relationship between which superpixels are visible and the resulting confidence scores, LIME estimates the importance of each region. 

The final visualization highlights the contours or pixels that most strongly contributed to the classification—such as the tip and shaft of an arrow—giving a clear, interpretable explanation of CLIP’s decision.