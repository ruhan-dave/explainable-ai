This project is inspired by various explainable AI techniques and innovations of the DeepSeek OCR paper, which utilized specialized ML models like SAM, CLIP, ViT in a pipeline to provide stronger analytical capabilities for document analysis. 

Released in late 2025, the DeepSeek OCR paper revolutionized how models process unstructured visual data by shifting from traditional text-first reading to a "vision-first" paradigm, where the model intrinsically understands the spatial and geometric relationships of a document before parsing its text. Optical Character Recognition (OCR) is the process of converting images of typed, handwritten, or printed text into machine-encoded text, serving as the foundational layer for modern document processing applications like automated invoice data extraction and digitized archival search. 

To help users and developers understand how DeepSeek OCR works, I've recreated the pipeline using open-source models (Meta's SAM, OpenAI's CLIP, etc), then applied XAI techniques (GradCam++, LIME analysis, bounding box visualization + confidence score ranking, etc) for different components of the OCR pipeline.

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

**Understanding the Core Models & XAI:**
* **CLIP (Contrastive Language-Image Pretraining):** Acts as the "brain" for zero-shot classification. It understands both images and text, allowing us to ask "Does this image patch look like an 'Arrow'?" without needing a custom-trained arrow detector.
* **SAM (Segment Anything Model):** Acts as the "eyes" for precise boundaries. It doesn't know *what* an object is, but it knows *where* it is and what its exact shape is, allowing us to isolate text boxes, flowchart lines, and structural blobs.
* **GradCAM++ (Gradient-weighted Class Activation Mapping):** An explainability technique that creates a heatmap over the image. It visualizes the model's attention, answering: *"When asked to find 'ViT', which specific pixels caused CLIP's neurons to fire the hardest?"*
* **LIME (Local Interpretable Model-agnostic Explanations):** An explainability technique that tests local features by perturbing them. By hiding parts of an image (like erasing the arrowhead) and seeing if the model still recognizes it, LIME proves exactly *which* features drove the classification.

The system runs through a 6-step pipeline:

### 1. Vision (Global Attention)
"We start with a 'Vision-First' approach. Before we try to read text or find boxes, we want to know: What grabs the model's attention? By running GradCAM on the CLIP encoder, we get this heat map. It proves the model isn't just guessing; it's actually attending to the relevant areas of the image based on our prompts."

The pipeline starts by asking: where does the model look when it searches for a given concept (e.g., for the DeepSeek OCR diagram, “SAM”, “ViT”)? We run GradCAM++ on CLIP, hooking into the final visual layers (`vision_model.encoder.layers...`). For each user-provided text prompt, the system computes a heatmap showing pixel-level activation intensity. This gives a global view of which regions of the image CLIP finds relevant to each concept, before we even talk about specific objects.

### 2. Segmentation (Object Discovery)
"Now we move to Object Discovery. Here, we use the Segment Anything Model (SAM) in a 'blind' mode. We aren't telling it what to look for, just to find everything that looks like a distinct object. We filter out the tiny noise and huge backgrounds to get a clean set of candidate regions."

Next, the system tries to discover all the distinct “things” in the image, without yet knowing what they are. Using SAM’s `AutomaticMaskGenerator`, we scan the entire image and produce binary masks representing candidate objects or regions. To keep the results meaningful, we filter out masks that are too small (noise) or too large (e.g., broad background regions). For efficiency, crop layers are explicitly disabled (`crop_n_layers=0`), which speeds up processing without sacrificing the quality we need for later steps.

### 3. Emergence (Classification)
"This is where the magic happens—Semantic Emergence. We take those blind segments and feed them into CLIP. We ask CLIP: 'Is this an Arrow? A Loop? A ViT block?' We only keep the matches where the model is confident. This effectively turns a raw image into a structured list of semantic elements."

Once we have a collection of anonymous segments, we need to give them semantic meaning. The system iterates over each mask from Step 2 and crops the original image to that mask’s bounding box. Each crop is fed into CLIP alongside a set of candidate labels: both user-specified prompts and common diagram elements such as “Arrow”, “Box”, or “Diagram”. CLIP performs zero-shot classification on each crop. If the top label’s confidence exceeds a threshold (0.25), that segment is considered “recognized” and is assigned that label.

### 4. Fusion (Visualization)
"Here we bring it all together. The Global Fusion step overlays our findings back onto the source image. This acts as our 'sanity check'—we can immediately see if the model has correctly understood the relationship between the labels and the visual diagram components."

To make the results understandable at a glance, the system overlays them onto the original image. For every recognized segment, we draw bounding boxes using OpenCV (`cv2.rectangle`) and render the predicted labels with `cv2.putText`. The output is a single, fused result image that shows all detected elements and their labels in context. This is what the user sees as the primary visual summary of the model’s understanding.

### 5. Structure Verification (Refinement)
"But what if we want to understand why a specific element was chosen? In this step, I select the 'Loop' element. We pass this specific region back into the SAM Predictor to get a pixel-perfect mask. This isolates the object completely from the rest of the diagram, setting the stage for deep analysis."

Often, a user wants to drill down into a single element—for example, clicking on a label like “ViT” to understand it more deeply. When a user selects a target, the system searches the classified segments from Step 3 for the best textual match. It then takes that segment’s bounding box and feeds it back into SAM, this time through the interactive SAM Predictor rather than the automatic generator. This produces a high-quality, precise segmentation mask focused on just that one object, giving us a clean region of interest for deeper analysis.

### 6. LIME Analysis (Deep Explanation)
"Finally, we have the Explanation. We use a custom implementation of LIME here. We break the object into 'superpixels' and flicker them on and off to see how CLIP's confidence changes. The green areas you see aren't just random highlights—they are the specific pixels that caused the model to say 'This is a Loop.' It's the ultimate proof of the model's reasoning."

Finally, the system explains why CLIP recognized that specific object as it did—for instance, why a particular region is considered an “arrow.” First, we extract the region of interest using the refined mask from Step 5. Inside this ROI, we run SLIC superpixel segmentation (`skimage.segmentation.slic`) to break the object into smaller, coherent subregions. LIME then perturbs these superpixels by randomly turning them on and off (e.g., graying some out) and generating many slightly modified versions of the ROI. For each perturbed image, we record CLIP’s confidence for the chosen label.

By fitting a linear regression model to the relationship between which superpixels are visible and the resulting confidence scores, LIME estimates the importance of each region. The final visualization highlights the contours or pixels that most strongly contributed to the classification—such as the tip and shaft of an arrow—giving a clear, interpretable explanation of CLIP’s decision.
