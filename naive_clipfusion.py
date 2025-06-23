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

# --- Main Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

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
image_paths = ["examples/room/images/no_overlap_2.jpg"]

# Load images and get their shape
images_for_vggt = load_and_preprocess_images(image_paths).to(DEVICE)
_, _, H, W = images_for_vggt.shape
print(f"Loaded images and derived shape: H={H}, W={W}")

# --- VGGT INFERENCE PROCEDURE ---
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=DTYPE):
        batched_images = images_for_vggt.unsqueeze(0)
        aggregated_tokens_list, ps_idx = vggt_model.aggregator(batched_images)
        pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
        vggt_extrinsics, vggt_intrinsics = pose_encoding_to_extri_intri(pose_enc, images_for_vggt.shape[-2:])
        depth_map, _ = vggt_model.depth_head(aggregated_tokens_list, batched_images, ps_idx)

# --- Construct 3D Points ---
point_cloud_3d = unproject_depth_map_to_point_map(
    depth_map.squeeze(0),
    vggt_extrinsics.squeeze(0),
    vggt_intrinsics.squeeze(0)
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
point_cloud_3d = point_cloud_3d[::10, :]
print(f"Reconstructed a point cloud with {point_cloud_3d.shape[0]} points.")

# --- 3. 2D FEATURE EXTRACTION (with CLIP) ---
print("\nExtracting 2D features with CLIP...")
pil_images = [Image.open(p) for p in image_paths]
image_inputs = clip_processor(images=pil_images, return_tensors="pt").to(DEVICE)

# Extract CLIP image features (these are in the shared embedding space)
with torch.no_grad():
    image_features = clip_model.get_image_features(pixel_values=image_inputs.pixel_values)

print(f"Extracted image features with shape: {image_features.shape}")

# --- 4. ROBUST FEATURE LIFTING (with PyTorch3D) ---
print("\nLifting 2D features to 3D point cloud...")
num_points = point_cloud_3d.shape[0]
num_views = len(pil_images)

# Initialize point features in CLIP embedding space
point_features_sum = torch.zeros(num_points, image_features.shape[1], device=DEVICE)
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

    u, v, z = projected_points[0, ..., 0], projected_points[0, ..., 1], projected_points[0, ..., 2]
    visible_mask = (z > 0) & (u.abs() < 1) & (v.abs() < 1)

    # Assign CLIP image features to visible points
    point_features_sum[visible_mask] += image_features[i]
    point_view_count[visible_mask] += 1

# Average the features
avg_point_features = point_features_sum / (point_view_count.unsqueeze(-1) + 1e-8)
print("Feature lifting complete.")

# --- 5. OPEN-VOCABULARY SEGMENTATION ---
print("\nPerforming zero-shot segmentation...")
text_query = "a red chair"

with torch.no_grad():
    text_inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True).to(DEVICE)
    text_features = clip_model.get_text_features(**text_inputs)

    # Normalize features for cosine similarity
    avg_point_features_norm = F.normalize(avg_point_features, p=2, dim=-1)
    text_features_norm = F.normalize(text_features, p=2, dim=-1)

    # Compute similarity scores
    similarity_scores = (avg_point_features_norm @ text_features_norm.T).squeeze()

    # Debug: Print similarity score statistics
    print(
        f"Similarity scores - Min: {similarity_scores.min():.4f}, Max: {similarity_scores.max():.4f}, Mean: {similarity_scores.mean():.4f}")

    # Use a more adaptive threshold or percentile-based approach
    segmentation_threshold = 0.15  # Lower threshold
    # Alternative: use top percentile
    # segmentation_threshold = torch.quantile(similarity_scores, 0.7).item()

    segmentation_mask = (similarity_scores > segmentation_threshold).cpu().numpy()
    num_segmented_points = segmentation_mask.sum()
    print(
        f"Segmentation complete. Found {num_segmented_points} points for query: '{text_query}' (threshold: {segmentation_threshold:.3f})")

# --- 6. VISUALIZATION (Conceptual Example) ---
try:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    # Here, we move to CPU and convert to numpy ONLY for visualization
    pcd.points = o3d.utility.Vector3dVector(point_cloud_3d.cpu().numpy())
    colors = np.full((num_points, 3), 0.5)
    colors[segmentation_mask] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("\nDisplaying 3D point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])
except ImportError:
    print("\nOpen3D not installed. Skipping visualization.")
    print("To visualize, run: pip install open3d")