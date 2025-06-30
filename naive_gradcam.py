import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn


# -------------- Helper Functions --------------#
def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]."""
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def getAttMap(img, attn_map, blur=True):
    """Create attention map overlay on image."""
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02 * max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1 * (1 - attn_map ** 0.7).reshape(attn_map.shape + (1,)) * img + \
               (attn_map ** 0.7).reshape(attn_map.shape + (1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, blur=True, save_path=None):
    """Visualize attention map."""
    # Set backend for headless environments
    import matplotlib
    current_backend = matplotlib.get_backend()
    if 'InterAgg' in current_backend or save_path is not None:
        matplotlib.use('Agg')  # Use non-interactive backend

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[1].imshow(getAttMap(img, attn_map, blur))
    axes[1].set_title("GradCAM Heatmap")
    for ax in axes:
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"GradCAM visualization saved to {save_path}")

    # Only show if not in a problematic backend
    try:
        if 'InterAgg' not in current_backend and save_path is None:
            plt.show()
        else:
            print("Skipping plt.show() due to backend compatibility")
    except Exception as e:
        print(f"Display skipped due to backend issue: {e}")

    plt.close()  # Clean up the figure


def load_image(img_path, resize=None):
    """Load and preprocess image."""
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def gradCAM_hf_clip(
        clip_model,
        clip_processor,
        image: Image.Image,
        text_query: str,
        layer_name: str = "last_layer",
        device: str = "cuda"
) -> torch.Tensor:
    """
    GradCAM for Hugging Face CLIP models (ViT-based).

    Args:
        clip_model: Hugging Face CLIPModel
        clip_processor: Hugging Face CLIPProcessor
        image: PIL Image
        text_query: Text query for GradCAM
        layer_name: Which layer to hook ('last_layer', 'second_last', or specific layer)
        device: Device to run on

    Returns:
        GradCAM heatmap as torch.Tensor
    """
    # Prepare inputs
    image_inputs = clip_processor(images=[image], return_tensors="pt").to(device)
    text_inputs = clip_processor(text=[text_query], return_tensors="pt").to(device)

    # Get the vision model and select layer to hook
    vision_model = clip_model.vision_model

    # For ViT, we can hook into different transformer layers
    if layer_name == "last_layer":
        target_layer = vision_model.encoder.layers[-1].layer_norm1
    elif layer_name == "second_last":
        target_layer = vision_model.encoder.layers[-2].layer_norm1
    elif layer_name == "pre_layernorm":
        target_layer = vision_model.pre_layrnorm  # Note: might be 'pre_layernorm' in some versions
    else:
        # Try to access custom layer
        try:
            target_layer = eval(f"vision_model.{layer_name}")
        except:
            print(f"Layer {layer_name} not found, using last layer")
            target_layer = vision_model.encoder.layers[-1].layer_norm1

    # Zero out any gradients
    pixel_values = image_inputs['pixel_values']
    if pixel_values.grad is not None:
        pixel_values.grad.data.zero_()

    pixel_values.requires_grad_(True)

    # Disable gradient computation for model parameters
    requires_grad = {}
    for name, param in clip_model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Hook into the target layer
    with Hook(target_layer) as hook:
        # Forward pass
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)

        # Get image features (this will trigger the hook)
        image_features = clip_model.get_image_features(pixel_values=pixel_values)

        # Compute similarity and backward pass
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
        similarity.backward()

        # Get gradients and activations
        gradients = hook.gradient
        activations = hook.activation

        if gradients is None or activations is None:
            print("Warning: No gradients captured. Trying alternative approach...")
            # Alternative: hook into an earlier layer
            target_layer_alt = vision_model.encoder.layers[-1]
            with Hook(target_layer_alt) as hook_alt:
                image_features = clip_model.get_image_features(pixel_values=pixel_values)
                similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                similarity.backward()
                gradients = hook_alt.gradient
                activations = hook_alt.activation

        if gradients is None:
            raise RuntimeError("Could not capture gradients. Try a different layer.")

        # Debug information
        print(f"Activations shape: {activations.shape}")
        print(f"Gradients shape: {gradients.shape}")

        # For ViT, the shape is typically [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = activations.shape

        # Handle different sequence lengths more robustly
        has_cls_token = True
        if seq_len == 197:  # Standard ViT-B/32 has 196 patches + 1 CLS token
            num_patches = 196
            patch_h = patch_w = 14  # 14x14 patches for ViT-B/32 on 224x224
        elif seq_len == 196:  # Already without CLS token
            num_patches = 196
            patch_h = patch_w = 14
            has_cls_token = False
        elif seq_len == 50:  # Smaller input, probably 7x7 patches + 1 CLS
            num_patches = 49
            patch_h = patch_w = 7
        elif seq_len == 49:  # 7x7 patches without CLS
            num_patches = 49
            patch_h = patch_w = 7
            has_cls_token = False
        else:
            # Try to infer patch grid size
            if seq_len > 1:
                # Check if it's a perfect square (without CLS)
                sqrt_seq = int(np.sqrt(seq_len))
                if sqrt_seq * sqrt_seq == seq_len:
                    num_patches = seq_len
                    patch_h = patch_w = sqrt_seq
                    has_cls_token = False
                else:
                    # Check if it's a perfect square + 1 (with CLS)
                    sqrt_seq_minus_1 = int(np.sqrt(seq_len - 1))
                    if sqrt_seq_minus_1 * sqrt_seq_minus_1 == (seq_len - 1):
                        num_patches = seq_len - 1
                        patch_h = patch_w = sqrt_seq_minus_1
                        has_cls_token = True
                    else:
                        # Fallback: use all tokens and estimate grid
                        num_patches = seq_len
                        patch_h = patch_w = int(np.sqrt(seq_len))
                        has_cls_token = False
                        print(f"Warning: Unusual sequence length {seq_len}, using estimated grid {patch_h}x{patch_w}")
            else:
                raise ValueError(f"Unexpected sequence length: {seq_len}")

        print(f"Detected: {num_patches} patches, {patch_h}x{patch_w} grid, CLS token: {has_cls_token}")

        # Remove CLS token if present
        if has_cls_token:
            activations = activations[:, 1:, :]  # Remove CLS token
            gradients = gradients[:, 1:, :]

        # Ensure we have the right number of patches
        assert activations.shape[
                   1] == num_patches, f"Mismatch: expected {num_patches} patches, got {activations.shape[1]}"

        # Compute GradCAM
        # Average gradients over the hidden dimension
        weights = gradients.mean(dim=-1, keepdim=True)  # [batch, patches, 1]

        # Weight the activations
        weighted_activations = (weights * activations).sum(dim=-1)  # [batch, patches]

        # Apply ReLU
        gradcam = F.relu(weighted_activations)

        print(f"GradCAM before reshape: {gradcam.shape}, target shape: [{batch_size}, {patch_h}, {patch_w}]")

        # Reshape to spatial dimensions
        gradcam = gradcam.reshape(batch_size, patch_h, patch_w)

        # Interpolate to input image size
        input_size = pixel_values.shape[-2:]  # [H, W]
        gradcam = F.interpolate(
            gradcam.unsqueeze(1),  # Add channel dimension
            size=input_size,
            mode='bicubic',
            align_corners=False
        )
        gradcam = gradcam.squeeze(1)  # Remove channel dimension

    # Restore gradient settings
    for name, param in clip_model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


def visualize_gradcam_hf_clip(
        clip_model,
        clip_processor,
        image_path: str,
        text_query: str,
        device: str = "cuda",
        save_path: str = None,
        layer_name: str = "last_layer",
        blur: bool = True
):
    """
    Complete workflow for GradCAM visualization with Hugging Face CLIP.

    Args:
        clip_model: Hugging Face CLIPModel
        clip_processor: Hugging Face CLIPProcessor
        image_path: Path to image file
        text_query: Text query for attention
        device: Device to run on
        save_path: Path to save visualization (optional)
        layer_name: Layer to hook into
        blur: Whether to blur the attention map
    """
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
        image_np = load_image(image_path, resize=224)  # Standard CLIP input size
    else:
        image = image_path  # Assume it's already a PIL Image
        image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    # Generate GradCAM
    gradcam = gradCAM_hf_clip(
        clip_model=clip_model,
        clip_processor=clip_processor,
        image=image,
        text_query=text_query,
        layer_name=layer_name,
        device=device
    )

    # Convert to numpy for visualization
    gradcam_np = gradcam.squeeze().detach().cpu().numpy()

    # Visualize
    viz_attn(image_np, gradcam_np, blur=blur, save_path=save_path)

    return gradcam_np


def get_available_layers(clip_model):
    """
    Helper function to explore available layers in the CLIP vision model.
    """
    print("Available layers in CLIP vision model:")
    vision_model = clip_model.vision_model

    print("Encoder layers:")
    for i, layer in enumerate(vision_model.encoder.layers):
        print(f"  Layer {i}: {type(layer).__name__}")
        for name, module in layer.named_children():
            print(f"    - {name}: {type(module).__name__}")

    print("\nOther vision model components:")
    for name, module in vision_model.named_children():
        if name != "encoder":
            print(f"  {name}: {type(module).__name__}")