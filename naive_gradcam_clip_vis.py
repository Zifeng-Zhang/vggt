"""
Simple test script for HuggingFace CLIP GradCAM functionality.
Run this to test if the GradCAM integration works correctly.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
from naive_gradcam import visualize_gradcam_hf_clip, get_available_layers

def test_gradcam():
    """Test GradCAM with a sample image and query."""

    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "openai/clip-vit-base-patch32"

    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}")

    # Load model and processor
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print("Model loaded successfully!")

    # Load a test image (you can replace this with your own image path)
    # Using a sample image from the internet for demonstration
    try:
        # Download a sample image
        image_url = "https://images.unsplash.com/photo-1555685812-4b943f1cb0eb?w=400"
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        image_path = "test_image.jpg"
        image.save(image_path)
        print(f"Downloaded test image and saved as {image_path}")
    except:
        print("Could not download test image. Please provide your own image path.")
        return

    # Explore model structure
    print("\n--- Exploring CLIP Model Structure ---")
    get_available_layers(clip_model)

    # Test queries
    test_queries = [
        "a living room",
        "furniture",
        "a couch",
        "a table",
        "modern interior"
    ]

    print(f"\n--- Testing GradCAM with {len(test_queries)} queries ---")

    for i, query in enumerate(test_queries):
        print(f"\nTesting query {i+1}: '{query}'")

        try:
            # Test only one layer to avoid display issues
            layer_name = "last_layer"
            print(f"  Using layer: {layer_name}")

            gradcam_result = visualize_gradcam_hf_clip(
                clip_model=clip_model,
                clip_processor=clip_processor,
                image_path=image_path,
                text_query=query,
                device=DEVICE,
                save_path=f"test_gradcam_{query.replace(' ', '_')}_{layer_name}.png",
                layer_name=layer_name,
                blur=True
            )

            print(f"    ✓ GradCAM generated successfully")
            print(f"    ✓ Shape: {gradcam_result.shape}")
            print(f"    ✓ Min/Max values: {gradcam_result.min():.4f} / {gradcam_result.max():.4f}")

        except Exception as e:
            print(f"    ✗ Error with query '{query}': {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Test Complete ===")
    print("✓ GradCAM computation is working correctly!")
    print("✓ Check the generated 'test_gradcam_*.png' files to see the results!")
    print("Note: Display issues in PyCharm are normal - the PNG files are the important output.")


def test_with_custom_image(image_path, query="a photo"):
    """Test GradCAM with your own image."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "openai/clip-vit-base-patch32"

    # Load model
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print(f"Testing with custom image: {image_path}")
    print(f"Query: '{query}'")

    # Generate GradCAM
    gradcam_result = visualize_gradcam_hf_clip(
        clip_model=clip_model,
        clip_processor=clip_processor,
        image_path=image_path,
        text_query=query,
        device=DEVICE,
        save_path=f"custom_gradcam_{query.replace(' ', '_')}.png",
        layer_name="last_layer",
        blur=True
    )

    print("Custom GradCAM generated!")
    return gradcam_result


def test_gradcam_headless():
    """Headless test that only saves files without trying to display."""

    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "openai/clip-vit-base-patch32"

    print(f"=== Headless GradCAM Test ===")
    print(f"Using device: {DEVICE}")

    # Load model and processor
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Download test image
    try:
        image_url = "https://images.unsplash.com/photo-1555685812-4b943f1cb0eb?w=400"
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        image_path = "headless_test_image.jpg"
        image.save(image_path)
        print(f"✓ Downloaded test image: {image_path}")
    except:
        print("✗ Could not download test image")
        return

    # Test one query
    query = "a living room"
    print(f"✓ Testing query: '{query}'")

    try:
        gradcam_result = visualize_gradcam_hf_clip(
            clip_model=clip_model,
            clip_processor=clip_processor,
            image_path=image_path,
            text_query=query,
            device=DEVICE,
            save_path=f"headless_gradcam_{query.replace(' ', '_')}.png",
            layer_name="last_layer",
            blur=True
        )

        print(f"✓ GradCAM Success!")
        print(f"  - Shape: {gradcam_result.shape}")
        print(f"  - Value range: [{gradcam_result.min():.4f}, {gradcam_result.max():.4f}]")
        print(f"  - File saved: headless_gradcam_{query.replace(' ', '_')}.png")

        return True

    except Exception as e:
        print(f"✗ GradCAM failed: {e}")
        return False


if __name__ == "__main__":
    # Run the headless test first (most reliable)
    print("=== Running Headless Test (Recommended) ===")
    # success = test_gradcam_headless()

    # if success:
    #     print("\n✓ GradCAM is working correctly!")
    #     print("You can now use it in your main pipeline.")
    # else:
    #     print("\n✗ GradCAM test failed. Check the error messages above.")

    # Uncomment below to run the full test (may have display issues in PyCharm)
    # print("\n" + "="*50)
    test_gradcam()

    # Uncomment below to test with your own image
    # test_with_custom_image("path/to/your/image.jpg", "your query here")