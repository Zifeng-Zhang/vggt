import torch
import numpy as np


def save_point_cloud_to_ply(points, filename="point_cloud.ply"):
    """
    Save point cloud to PLY format for visualization in MeshLab

    Args:
        points: torch.Tensor of shape (N, 3) containing 3D points
        filename: output filename
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points

    num_points = points_np.shape[0]

    # Write PLY header
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write point coordinates
        for i in range(num_points):
            f.write(f"{points_np[i, 0]:.6f} {points_np[i, 1]:.6f} {points_np[i, 2]:.6f}\n")

    print(f"Point cloud saved to {filename}")


def save_colored_point_cloud_to_ply(points, segmentation_mask, filename="colored_point_cloud.ply"):
    """
    Save point cloud with segmentation colors to PLY format

    Args:
        points: torch.Tensor of shape (N, 3) containing 3D points
        segmentation_mask: numpy array of shape (N,) with boolean segmentation
        filename: output filename
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points

    num_points = points_np.shape[0]

    # Create colors: red for segmented points, gray for others
    colors = np.full((num_points, 3), [128, 128, 128])  # Gray
    colors[segmentation_mask] = [255, 0, 0]  # Red for segmented points

    # Write PLY header
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write point coordinates and colors
        for i in range(num_points):
            f.write(f"{points_np[i, 0]:.6f} {points_np[i, 1]:.6f} {points_np[i, 2]:.6f} "
                    f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")

    print(f"Colored point cloud saved to {filename}")


def extract_dense_clip_features(images, clip_model, clip_processor, device):
    """
    Extract dense CLIP features using a better approach
    """
    dense_features_list = []
    global_features_list = []

    for img in images:
        inputs = clip_processor(images=[img], return_tensors="pt").to(device)

        with torch.no_grad():
            # Get both global and patch features
            vision_outputs = clip_model.vision_model(pixel_values=inputs.pixel_values)

            # Global image features (properly projected)
            global_features = clip_model.get_image_features(pixel_values=inputs.pixel_values)
            global_features_list.append(global_features)

            # Patch features
            patch_features = vision_outputs.last_hidden_state.squeeze(0)  # (num_patches, 768)
            patch_features = patch_features[1:]  # Remove CLS token

            # Apply the same projection that CLIP uses for final features
            # Fix: Use the weight matrix from the Linear layer
            if hasattr(clip_model, 'visual_projection') and clip_model.visual_projection is not None:
                # Get the weight matrix from the Linear layer
                projection_weight = clip_model.visual_projection.weight  # (512, 768)
                projected_patches = torch.mm(patch_features, projection_weight.T)  # (49, 512)
            elif hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
                # Fallback: use the text projection weight
                projection_weight = clip_model.text_projection.weight  # (512, 768)
                projected_patches = torch.mm(patch_features, projection_weight.T)  # (49, 512)
            else:
                # Last resort: learn a simple projection
                print("Warning: No projection found, using learned projection")
                if not hasattr(extract_dense_clip_features, 'learned_projection'):
                    extract_dense_clip_features.learned_projection = torch.nn.Linear(768, 512).to(device)
                    # Initialize to approximate CLIP's projection
                    torch.nn.init.xavier_uniform_(extract_dense_clip_features.learned_projection.weight)

                projected_patches = extract_dense_clip_features.learned_projection(patch_features)

            # Reshape to spatial grid
            patch_size = int(np.sqrt(projected_patches.shape[0]))
            spatial_features = projected_patches.reshape(patch_size, patch_size, -1)

            dense_features_list.append(spatial_features)

    return dense_features_list, global_features_list


def debug_clip_model_structure(clip_model):
    """
    Debug function to understand CLIP model structure
    """
    print("=== CLIP Model Structure Debug ===")
    print("Available attributes:")
    for attr in dir(clip_model):
        if not attr.startswith('_') and not callable(getattr(clip_model, attr)):
            try:
                value = getattr(clip_model, attr)
                print(f"  {attr}: {type(value)}")
                if hasattr(value, 'weight'):
                    print(f"    - weight shape: {value.weight.shape}")
            except:
                print(f"  {attr}: <unable to access>")

    # Check specific projections
    if hasattr(clip_model, 'visual_projection'):
        print(f"visual_projection: {type(clip_model.visual_projection)}")
        if clip_model.visual_projection is not None:
            print(f"  - weight shape: {clip_model.visual_projection.weight.shape}")

    if hasattr(clip_model, 'text_projection'):
        print(f"text_projection: {type(clip_model.text_projection)}")
        if clip_model.text_projection is not None:
            print(f"  - weight shape: {clip_model.text_projection.weight.shape}")

    print("=== End Debug ===\n")


def visualize_image_features(image_features, original_image, save_path="image_features_visualization.png"):
    """
    Visualize CLIP image features by creating a heatmap overlay on the original image.
    Since CLIP features are global, we'll create a simple feature magnitude visualization.

    Args:
        image_features: torch.Tensor of shape (1, feature_dim) - CLIP image features
        original_image: PIL Image - the original input image
        save_path: str - path to save the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Convert features to numpy
    features_np = image_features.cpu().numpy().squeeze()  # Shape: (feature_dim,)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Feature magnitude histogram
    axes[0, 1].hist(features_np, bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_title("Feature Value Distribution")
    axes[0, 1].set_xlabel("Feature Value")
    axes[0, 1].set_ylabel("Frequency")

    # Feature values as a bar plot (first 50 dimensions)
    axes[1, 0].bar(range(min(50, len(features_np))), features_np[:50])
    axes[1, 0].set_title("First 50 Feature Dimensions")
    axes[1, 0].set_xlabel("Feature Dimension")
    axes[1, 0].set_ylabel("Feature Value")

    # Feature magnitude heatmap (reshape features into a square-ish grid for visualization)
    feature_dim = len(features_np)
    # Calculate grid size that can accommodate all features
    grid_size = int(np.ceil(np.sqrt(feature_dim)))

    # Create padded features array with the correct size
    padded_features = np.zeros(grid_size * grid_size)
    padded_features[:feature_dim] = features_np  # This should work now

    feature_grid = padded_features.reshape(grid_size, grid_size)
    im = axes[1, 1].imshow(feature_grid, cmap='viridis', aspect='auto')
    axes[1, 1].set_title("Feature Values as 2D Grid")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Image features visualization saved to {save_path}")

    # Also save feature statistics
    stats_text = f"""
Image Feature Statistics:
- Feature dimension: {feature_dim}
- Min value: {features_np.min():.6f}
- Max value: {features_np.max():.6f}
- Mean value: {features_np.mean():.6f}
- Std deviation: {features_np.std():.6f}
- L2 norm: {np.linalg.norm(features_np):.6f}
"""

    with open("image_features_stats.txt", "w") as f:
        f.write(stats_text)

    print("Feature statistics saved to image_features_stats.txt")
    return features_np
