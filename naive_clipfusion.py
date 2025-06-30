import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# --- Library Imports ---
from transformers import CLIPProcessor, CLIPModel
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
# Import the new function for camera pose decoding
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# PyTorch3D imports for robust projection
from pytorch3d.renderer import PerspectiveCameras

from naive_utils import *
from clip_gradcam import gradCAM


# --- Main Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

IMAGE_FOLDER = "examples/room/images"
all_files = os.listdir(IMAGE_FOLDER)
image_extensions = ('.png', '.jpg', '.jpeg')


# --- 1. MODEL LOADING ---

# Load VGGT Model
print("Loading VGGT model...")
vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)

# Load Local Hugging Face CLIP Model
print("Loading local CLIP model...")
# NOTE: Replace this path with the correct path to your folder containing 'pytorch_model.bin', 'config.json', etc.
LOCAL_CLIP_PATH = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(LOCAL_CLIP_PATH).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_PATH)
print("All models loaded successfully!")


# --- 2. 3D SCENE RECONSTRUCTION (with VGGT) ---
print("\nReconstructing scene with VGGT...")
# Replace with your image paths
image_list = [
    os.path.join(IMAGE_FOLDER, f)
    for f in all_files
    if f.lower().endswith(image_extensions)
]


# Load images and get their shape
images_for_vggt = load_and_preprocess_images(image_list).to(DEVICE)
_, _, H, W = images_for_vggt.shape
print(f"Loaded images and derived shape: H={H}, W={W}")

# --- VGGT INFERENCE PROCEDURE ---
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=DTYPE):
        batched_images = images_for_vggt.unsqueeze(0)
        aggregated_tokens_list, ps_idx = vggt_model.aggregator(batched_images)

        # Predict Cameras
        pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]

        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        vggt_extrinsics, vggt_intrinsics = pose_encoding_to_extri_intri(pose_enc, images_for_vggt.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = vggt_model.depth_head(aggregated_tokens_list, batched_images, ps_idx)

        # Predict Point Maps
        point_map, point_conf = vggt_model.point_head(aggregated_tokens_list, batched_images, ps_idx)


# --- Construct 3D Points from Depth Maps and Cameras---
point_cloud_3d = unproject_depth_map_to_point_map(
    depth_map.cpu().numpy().squeeze(0),
    vggt_extrinsics.cpu().numpy().squeeze(0),
    vggt_intrinsics.cpu().numpy().squeeze(0)
)

# Convert to PyTorch tensor if it's a NumPy array and move to device
if isinstance(point_cloud_3d, np.ndarray):
    point_cloud_3d = torch.from_numpy(point_cloud_3d).float().to(DEVICE)
elif not isinstance(point_cloud_3d, torch.Tensor):
    point_cloud_3d = torch.tensor(point_cloud_3d).float().to(DEVICE)
else:
    point_cloud_3d = point_cloud_3d.float().to(DEVICE)

# Reshape from (H*W, 3) to a more usable list of points
point_cloud_3d = point_cloud_3d.reshape(-1, 3)

# Optional: Downsample point cloud for faster processing if it's too dense
# point_cloud_3d = point_cloud_3d[::10, :]

print(f"Reconstructed a point cloud with {point_cloud_3d.shape[0]} points.")
# Call this function after you create point_cloud_3d
save_point_cloud_to_ply(point_cloud_3d, "reconstructed.ply")


# --- 3. 2D FEATURE EXTRACTION (with CLIP) ---
print("\nExtracting 2D features with CLIP...")
pil_images = [Image.open(p) for p in image_list]
image_inputs = clip_processor(images=pil_images, return_tensors="pt").to(DEVICE)

# debug_clip_model_structure(clip_model)

dense_features_list, global_features_list = extract_dense_clip_features(pil_images, clip_model, clip_processor, device=DEVICE)
print(f"Extracted dense features: {len(dense_features_list)} images, each with shape {dense_features_list[0].shape}")


# Visualize the extracted image features
image_caption = 'a bed'
text_inputs = clip_processor(text=[image_caption], return_tensors="pt").to(DEVICE)

#TODO: Integrate gradCAM here to visualize CLIP attention map



# --- 4. ROBUST FEATURE LIFTING (with PyTorch3D) ---
print("\nLifting 2D features to 3D point cloud...")
num_points = point_cloud_3d.shape[0]
num_views = len(pil_images)

# Initialize point features in CLIP embedding space
point_features_sum = torch.zeros(num_points, 512, device=DEVICE)  # Changed to 512
point_view_count = torch.zeros(num_points, device=DEVICE)

vggt_extrinsics_squeezed = vggt_extrinsics.squeeze(0)
vggt_intrinsics_squeezed = vggt_intrinsics.squeeze(0)

for i in range(num_views):
    R = vggt_extrinsics_squeezed[i, :3, :3].unsqueeze(0)
    T = vggt_extrinsics_squeezed[i, :3, 3].unsqueeze(0)

    # Convert 3x3 intrinsic matrix to 4x4 format expected by PyTorch3D
    K_3x3 = vggt_intrinsics_squeezed[i]
    K_4x4 = torch.zeros(4, 4, device=DEVICE, dtype=K_3x3.dtype)
    K_4x4[:3, :3] = K_3x3
    K_4x4[3, 3] = 1.0
    K = K_4x4.unsqueeze(0)

    cameras = PerspectiveCameras(device=DEVICE, R=R, T=T, K=K, image_size=((H, W),))

    # Project 3D points to 2D
    projected_points = cameras.transform_points(point_cloud_3d.unsqueeze(0))

    # Convert to pixel coordinates
    u, v, z = projected_points[0, ..., 0], projected_points[0, ..., 1], projected_points[0, ..., 2]

    # Convert from normalized coordinates [-1, 1] to pixel coordinates
    pixel_u = ((u + 1) * W / 2).long()
    pixel_v = ((v + 1) * H / 2).long()

    # Visibility mask
    visible_mask = (z > 0) & (pixel_u >= 0) & (pixel_u < W) & (pixel_v >= 0) & (pixel_v < H)

    # Map pixel coordinates to patch coordinates
    patch_size = dense_features_list[i].shape[0]  # 7 for ViT-B/32
    patch_u = (pixel_u * patch_size / W).long().clamp(0, patch_size - 1)
    patch_v = (pixel_v * patch_size / H).long().clamp(0, patch_size - 1)

    # Get spatial features for visible points
    spatial_features = dense_features_list[i].to(DEVICE)  # (7, 7, 512)

    for j in range(num_points):
        if visible_mask[j]:
            # Get the corresponding patch feature (now already in 512-dim space)
            patch_feature = spatial_features[patch_v[j], patch_u[j]]  # (512,)
            point_features_sum[j] += patch_feature
            point_view_count[j] += 1

# Average features and project to CLIP embedding space
avg_point_features = point_features_sum / (point_view_count.unsqueeze(-1) + 1e-8)
print("Feature lifting complete.")


# --- 5. OPEN-VOCABULARY SEGMENTATION ---
print("\nPerforming zero-shot segmentation...")
text_query = "a photo of a chair"

with torch.no_grad():
    text_inputs = clip_processor(text=[text_query], return_tensors="pt").to(DEVICE)
    text_features = clip_model.get_text_features(**text_inputs)

    # Now both features are in the same space!
    point_features_norm = F.normalize(avg_point_features, p=2, dim=-1)
    text_features_norm = F.normalize(text_features, p=2, dim=-1)

    similarity_scores = (point_features_norm @ text_features_norm.T).squeeze()

    # Debug: Check for points with valid features
    valid_points = (point_view_count > 0)
    print(f"Points with valid features: {valid_points.sum()}")

    print(
        f"Similarity scores - Min: {similarity_scores.min():.4f}, Max: {similarity_scores.max():.4f}, Mean: {similarity_scores.mean():.4f}")

    # Use percentile-based threshold
    percentile_threshold = torch.quantile(similarity_scores[valid_points], 0.7).item()
    absolute_threshold = 0.15  # Reasonable threshold for CLIP similarities

    segmentation_threshold = max(percentile_threshold, absolute_threshold)
    segmentation_mask = (similarity_scores > segmentation_threshold).cpu().numpy()

    num_segmented_points = segmentation_mask.sum()

    print(
        f"Segmentation complete. Found {num_segmented_points} points for query: '{text_query}' (threshold: {segmentation_threshold:.3f})")

    save_colored_point_cloud_to_ply(point_cloud_3d, segmentation_mask, "segmented_point_cloud.ply")
