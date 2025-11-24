import os
# Core libraries
import cv2
import torch
import numpy as np
import base64
import io
import matplotlib
# Force headless backend for Matplotlib (Crucial for Docker/Cloud)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse # Added FileResponse

# Analysis Tools
from sklearn.linear_model import LinearRegression
from skimage.segmentation import slic, mark_boundaries

# AI Models
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from PIL import Image

# --- UTILITIES ---

def gradcampp_clip_single(model, pixel_values, input_ids, text_index, layer_path, image_size):
    # Compute GradCAM++ heatmap for a specific CLIP layer and text prompt
    activations = {}
    gradients = {}

    module = model
    try:
        for name in layer_path.split("."):
            if name.isdigit():
                module = module[int(name)]
            else:
                module = getattr(module, name)
    except (AttributeError, IndexError) as e:
        return np.zeros(image_size, dtype=np.float32)

    def forward_hook(_, __, output):
        activations["value"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    def backward_hook(_, __, grad_out):
        gradients["value"] = grad_out[0].detach() if isinstance(grad_out, tuple) else grad_out.detach()

    f_handle = module.register_forward_hook(forward_hook)
    b_handle = module.register_full_backward_hook(backward_hook)

    pixel_values = pixel_values.requires_grad_(True)
    outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    logits_per_image = outputs.logits_per_image
    score = logits_per_image[0, text_index]

    model.zero_grad()
    score.backward(retain_graph=False)

    f_handle.remove()
    b_handle.remove()

    token_act = activations["value"]
    token_grad = gradients["value"]

    if token_act.dim() == 3: token_act = token_act.squeeze(0)
    if token_grad.dim() == 3: token_grad = token_grad.squeeze(0)

    if token_act.shape[0] > 1:
        token_act = token_act[1:, :]
        token_grad = token_grad[1:, :]
    else:
        return np.zeros(image_size, dtype=np.float32)

    num_patches, hidden = token_act.shape
    grid = int(num_patches ** 0.5)
    
    act_map = token_act.T.reshape(hidden, grid, grid)
    grad_map = token_grad.T.reshape(hidden, grid, grid)

    grad_2 = grad_map ** 2
    grad_3 = grad_map ** 3
    denom = 2 * grad_2 + act_map * grad_3 + 1e-8
    denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
    alpha = grad_2 / denom

    relu_grad = torch.relu(grad_map)
    weights = (alpha * relu_grad).sum(dim=(1, 2))

    cam = (weights.unsqueeze(-1).unsqueeze(-1) * act_map).sum(dim=0)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    return cam

def run_clip_gradcam(model, processor, image, text_prompts, target_text_idx, layers=None):    
    # Run GradCAM++ across multiple CLIP layers and average results
    layers = [
            "vision_model.encoder.layers.2",
            "vision_model.encoder.layers.4",
            "vision_model.encoder.layers.6",
            "vision_model.encoder.layers.8",
            "vision_model.encoder.layers.10",
            "vision_model.encoder.layers.11"
        ]
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(model.device)
    input_ids = inputs["input_ids"].to(model.device)
    image_size = (image.size[1], image.size[0])

    per_layer_maps = []
    for layer_path in layers:
        try:
            layer_map = gradcampp_clip_single(
                model=model, pixel_values=pixel_values, input_ids=input_ids,
                text_index=target_text_idx, layer_path=layer_path, image_size=image_size
            )
            if isinstance(layer_map, np.ndarray) and layer_map.ndim == 2:
                per_layer_maps.append(layer_map.astype(np.float32))
        except: continue

    if not per_layer_maps: return np.zeros(image_size, dtype=np.float32), []
    
    stacked = np.stack(per_layer_maps, axis=0)
    combined = stacked.mean(axis=0)
    combined = np.nan_to_num(combined)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    return combined, per_layer_maps

def show_cam_on_image(img_np, heatmap, alpha=0.4):
    # Overlay GradCAM heatmap on original image
    if img_np.dtype != np.uint8: img_np = (img_np * 255).astype(np.uint8)
    heatmap = np.clip(heatmap, 0, 1)
    if heatmap.shape != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)

# --- CLASSES ---

class StructureAwareLIME:
    # LIME explainer that perturbs image segments to explain CLIP predictions
    def __init__(self, clip_model, clip_processor, device):
        self.model = clip_model
        self.processor = clip_processor
        self.device = device

    def run_structural_explanation(self, image_rgb, struct_mask, target_text, n_samples=100):
        # Extract ROI from structural mask and run LIME on segments
        rows, cols = np.where(struct_mask)
        y1, y2 = np.min(rows), np.max(rows)
        x1, x2 = np.min(cols), np.max(cols)
        pad = 30
        y1, y2 = max(0, y1-pad), min(image_rgb.shape[0], y2+pad)
        x1, x2 = max(0, x1-pad), min(image_rgb.shape[1], x2+pad)
        roi = image_rgb[y1:y2, x1:x2]

        segments = slic(roi, n_segments=50, compactness=10, sigma=1, start_label=1)
        perturbations = []
        num_segments = np.max(segments)
        mask_matrix = np.random.binomial(1, 0.5, size=(n_samples, num_segments))
        mask_matrix[0, :] = 1 

        for i in range(n_samples):
            temp_img = roi.copy()
            zeros = np.where(mask_matrix[i] == 0)[0]
            mask = np.isin(segments, zeros + 1)
            temp_img[mask] = (temp_img[mask] * 0.2 + 128 * 0.8).astype(np.uint8)
            perturbations.append(temp_img)

        inputs = self.processor(text=[target_text], images=perturbations, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits_per_image.cpu().numpy().flatten()

        reg = LinearRegression()
        reg.fit(mask_matrix, scores)
        return roi, segments, reg.coef_

class DeepSeekMimicOCR:
    # Main pipeline class for explainable OCR using CLIP and SAM
    def __init__(self):
        # Intelligent Device Selection
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.sam_device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'      # Fast for CLIP
            self.sam_device = 'cpu'  # Stable for SAM (Avoids float64 crash)
        else:
            self.device = 'cpu'
            self.sam_device = 'cpu'
        
        print(f"Init: CLIP on {self.device}, SAM on {self.sam_device}")
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        sam_checkpoint = "sam_vit_b_01ec64.pth"
        if not os.path.exists(sam_checkpoint):
             raise RuntimeError(f"SAM Checkpoint {sam_checkpoint} not found! Check Dockerfile.")
        
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.sam_device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam, 
            
            # OPTIMIZATION SETTINGS:
            points_per_side=16,          # Was 32. Reduces calculation
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            
            crop_n_layers=0,             # Was 1. Disables "Zoom in and re-run", massive speedup.
            
            crop_n_points_downscale_factor=2, 
            min_mask_region_area=100,
        )
        self.predictor = SamPredictor(sam)
        print("Pipeline initialized.")

    def preprocess(self, image_bytes):
        # Decode image bytes to RGB array, resize if large
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Invalid Image")
        h, w = image.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- APP & ROUTES ---

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

session_data = {"image": None, "pil_image": None, "refined_masks": [], "emerged_tokens": [], "pipeline_log": {}, "verified_mask": None}
# Global session state for pipeline steps
pipeline = None

@app.on_event("startup")
def load_models():
    # Load AI models on app start
    global pipeline
    pipeline = DeepSeekMimicOCR()

# -- SERVE FRONTEND --
@app.get("/")
async def read_index():
    # Serve the main HTML frontend
    return FileResponse('index.html')

def plt_to_base64():
    # Convert matplotlib figure to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def numpy_to_base64(img_array):
    # Convert numpy image array to base64
    pil_img = Image.fromarray(img_array)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.post("/upload")
async def upload(file: UploadFile = File(...), prompts: str = Form(None)):
    # Upload image and set up session for pipeline
    contents = await file.read()
    try:
        image = pipeline.preprocess(contents)
    except Exception as e: raise HTTPException(400, str(e))
    
    session_data.update({"image": image, "pil_image": Image.fromarray(image), "emerged_tokens": [], "verified_mask": None})
    cand = [p.strip() for p in prompts.split(',')] if prompts else ["SAM", "ViT", "Encoder"]
    session_data["pipeline_log"] = {'pipeline_params': {'candidate_prompts': cand}}
    return {"message": "Ready"}

@app.post("/step1_vision")
async def step1():
    # Step 1: Generate GradCAM heatmaps for each prompt
    if session_data["image"] is None: raise HTTPException(400, "No image")
    prompts = session_data["pipeline_log"]['pipeline_params']['candidate_prompts']
    pil = session_data["pil_image"]
    heatmaps = {}
    for i, p in enumerate(prompts):
        hm, _ = run_clip_gradcam(pipeline.clip_model, pipeline.clip_processor, pil, prompts, i)
        heatmaps[p] = {"overlay": numpy_to_base64(show_cam_on_image(np.array(pil), hm, 0.25))}
    return {"heatmaps": heatmaps}

@app.post("/step2_segmentation")
async def step2():
    # Step 2: Generate and filter SAM masks
    if session_data["image"] is None: raise HTTPException(400, "No image")
    masks = pipeline.mask_generator.generate(session_data["image"])
    refined = [m for m in masks if 500 < m['area'] < (session_data["image"].size * 0.5)]
    session_data["refined_masks"] = refined
    vis = session_data["image"].copy()
    for ann in sorted(refined, key=lambda x: x['area'], reverse=True):
        m = ann['segmentation']
        vis[m] = vis[m] * 0.5 + np.concatenate([np.random.random(3), [0.35]])[:3] * 255 * 0.5
    return {"visualization": numpy_to_base64(vis.astype(np.uint8))}

@app.post("/step3_emergence")
async def step3():
    # Step 3: Classify mask crops with CLIP to find elements
    if session_data["image"] is None: raise HTTPException(400, "No image")
    labels = session_data["pipeline_log"]['pipeline_params']['candidate_prompts'] + ["Arrow", "Box", "Diagram"]
    elements = []
    for i, m in enumerate(session_data["refined_masks"]):
        x, y, w, h = m['bbox']
        crop = session_data["pil_image"].crop((x, y, x+w, y+h))
        inp = pipeline.clip_processor(text=labels, images=crop, return_tensors="pt", padding=True).to(pipeline.device)
        with torch.no_grad():
            score, idx = pipeline.clip_model(**inp).logits_per_image.softmax(dim=1).max(dim=1)
        if score.item() > 0.25:
            elements.append({"id": i, "bbox": [x,y,w,h], "predicted_token": labels[idx], "confidence": round(score.item(),3)})
    session_data['emerged_tokens'] = elements
    return {"emerged_elements": elements}

@app.post("/step4_fusion")
async def step4():
    # Step 4: Visualize detected elements on image
    vis = np.array(session_data["pil_image"]).copy()
    for e in session_data.get('emerged_tokens', []):
        x,y,w,h = [int(v) for v in e['bbox']]
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(vis, e['predicted_token'], (x,y-5), 0, 0.5, (0,0,0), 1)
    return {"final_image": numpy_to_base64(vis)}

@app.post("/step5_structure_verification")
async def step5(target_text: str = Form("SAM")):
    # Step 5: Verify target element with SAM predictor
    cands = [e for e in session_data['emerged_tokens'] if target_text.lower() in e['predicted_token'].lower()]
    if not cands: return {"error": "Target not found"}
    best = sorted(cands, key=lambda x: x['confidence'], reverse=True)[0]
    x,y,w,h = best['bbox']
    
    pipeline.predictor.set_image(session_data["image"])
    masks, scores, _ = pipeline.predictor.predict(box=np.array([x,y,x+w,y+h])[None, :], multimask_output=True)
    struct_mask = masks[np.argmax(scores)]
    session_data.update({'verified_mask': struct_mask, 'verified_bbox': [x,y,w,h]})
    
    plt.figure(figsize=(6,4))
    plt.imshow(session_data["image"])
    overlay = np.zeros((*struct_mask.shape[-2:], 4))
    overlay[struct_mask, :3] = [0,1,0]
    overlay[struct_mask, 3] = 0.5
    plt.imshow(overlay)
    plt.gca().add_patch(plt.Rectangle((x,y), w, h, linewidth=2, edgecolor='blue', facecolor='none'))
    plt.axis('off')
    return {"visualization": plt_to_base64()}

@app.post("/step6_lime_analysis")
async def step6(target_text: str = Form("SAM")):
    # Step 6: Run LIME explanation on verified structure
    if session_data["verified_mask"] is None: return {"error": "Run Step 5 first"}
    explainer = StructureAwareLIME(pipeline.clip_model, pipeline.clip_processor, pipeline.device)
    roi, segs, weights = explainer.run_structural_explanation(session_data["image"], session_data["verified_mask"], target_text)
    
    heatmap = np.zeros(roi.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights): heatmap[segs == (i+1)] = w
    heatmap = heatmap / (np.max(np.abs(weights)) + 1e-9)
    
    overlay = np.zeros_like(roi, dtype=np.uint8)
    overlay[heatmap > 0, 1] = np.clip(heatmap[heatmap > 0] * 255 * 1.5, 0, 255)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(roi); ax[0].set_title("Component")
    ax[1].imshow(mark_boundaries(roi, segs)); ax[1].set_title("Segments")
    ax[2].imshow(cv2.addWeighted(roi, 0.6, overlay, 0.6, 0)); ax[2].set_title("Influence")
    for a in ax: a.axis('off')
    return {"visualization": plt_to_base64()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)