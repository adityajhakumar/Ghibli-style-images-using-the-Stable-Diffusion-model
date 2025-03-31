# Ghibli-style-images-using-the-Stable-Diffusion-model



### Imports
```python
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
from google.colab import files
import matplotlib.pyplot as plt
import time
```
- **diffusers**: This library provides the `StableDiffusionImg2ImgPipeline` class, which is used to perform image-to-image transformations using the Stable Diffusion model.
- **torch**: PyTorch library for tensor computations and GPU acceleration.
- **PIL (Python Imaging Library)**: Used for opening, manipulating, and saving images.
- **io**: Provides Python's core tools for working with streams (used here for handling uploaded files).
- **google.colab.files**: A module from Google Colab that allows you to upload and download files easily.
- **matplotlib.pyplot**: A plotting library used to display images in a visual format.
- **time**: Used for measuring the time taken to generate images.

### Load Model Function
```python
def load_model():
    model_id = "nitrosocke/Ghibli-Diffusion"  # Correct model ID
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("Loading model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()  # Optimize memory usage
    print("Model loaded!")
    return pipe
```
- This function loads the Ghibli-style model from Hugging Face's model hub using the specified `model_id`.
- It checks if a GPU is available and sets the appropriate data type (`float16` for GPU and `float32` for CPU) to optimize performance.
- The model is moved to the appropriate device (GPU or CPU) and attention slicing is enabled to reduce memory usage during inference.

### Generate Ghibli Image Function
```python
def generate_ghibli_image(image, pipe, strength):
    image = image.convert("RGB")
    image = image.resize((512, 512))  # Ensure proper size
    
    prompt = "Ghibli-style anime painting, soft pastel colors, highly detailed, masterpiece"
    
    print("Generating image...")
    start_time = time.time()
    result = pipe(prompt=prompt, image=image, strength=strength).images[0]
    print(f"Image generated in {time.time() - start_time:.2f} seconds!")
    return result
```
- This function takes an input image, the pipeline object (`pipe`), and a `strength` parameter that controls how much stylization is applied.
- The input image is converted to RGB format and resized to 512x512 pixels (the expected input size for the model).
- A prompt describing the desired output style is defined.
- The function uses the pipeline to generate a new image based on the input image and prompt. The time taken for generation is printed.

### Check for GPU Availability
```python
gpu_info = "GPU is available!" if torch.cuda.is_available() else "Warning: GPU not available. Processing will be slow."
print(gpu_info)
```
- This checks whether a GPU is available and prints a message accordingly. If no GPU is available, it warns that processing may be slow.

### Main Execution Flow
```python
# Load the model
pipe = load_model()

# Upload image section
print("Please upload your image file:")
uploaded = files.upload()

if uploaded:
    file_name = list(uploaded.keys())[0]
    image = Image.open(io.BytesIO(uploaded[file_name]))
    
    # Display original image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    # Ask for strength input with error handling
    while True:
        try:
            strength = float(input("Enter stylization strength (0.3-0.8, recommended 0.6): "))
            strength = max(0.3, min(0.8, strength))  # Clamp between 0.3 and 0.8
            break
        except ValueError:
            print("Invalid input. Please enter a number between 0.3 and 0.8.")
    
    # Generate and display the result
    result_img = generate_ghibli_image(image, pipe, strength)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(result_img)
    plt.title("Ghibli Portrait")
    plt.axis('off')
    plt.show()
    
    # Save the output image and offer download
    output_filename = f"ghibli_portrait_{file_name}"
    result_img.save(output_filename)
    files.download(output_filename)
    print(f"Image saved as {output_filename} and download initiated!")
else:
    print("No file was uploaded. Please run the cell again and upload an image.")
```
1. **Load Model**: The Ghibli-style model is loaded using the `load_model` function.
2. **Upload Image**: Users are prompted to upload an image file using Google Colab's file upload functionality.
3. **Display Original Image**: The uploaded image is displayed using Matplotlib.
4. **Input Strength**: Users are prompted to enter a stylization strength between 0.3 and 0.8 (recommended value is around 0.6). Error handling ensures valid input.
5. **Generate Image**: The `generate_ghibli_image` function generates a Ghibli-style portrait based on the uploaded image and specified strength.
6. **Display Result**: The generated Ghibli-style image is displayed.
7. **Save and Download**: The resulting image is saved locally, and a download link is provided.

## Suggested README File Format

Here's a suggested README.md file format that you can use:

```markdown
# Ghibli Style Image Generator

This project utilizes a pre-trained Ghibli-style diffusion model to transform your images into beautiful Ghibli-style artwork.

## Requirements

- Python 3.x
- Google Colab (for easy execution)
- Libraries:
  - diffusers
  - torch (with CUDA support recommended)
  - Pillow
  - matplotlib

## Installation

You can install the required libraries using pip:

```
pip install diffusers torch torchvision Pillow matplotlib
```

## Usage

1. Open this notebook in Google Colab.
2. Run all cells sequentially.
3. When prompted, upload an image file (JPG or PNG).
4. Enter a stylization strength between `0.3` and `0.8` when prompted (recommended value: `0.6`).
5. The original and generated images will be displayed.
6. The generated Ghibli-style portrait will be saved locally and automatically downloaded.

## Example

![Original Image](path_to_original_image_example.jpg)  
*Original Image*

![Ghibli Portrait](path_to_generated_image_example.jpg)  
*Generated Ghibli Portrait*

## Notes

- Ensure you have access to a GPU in Google Colab for faster processing.
- Adjust the strength parameter to see different levels of stylization.

## License

This project is licensed under [MIT License](LICENSE).
```

### Summary

This code provides an interactive way to transform images into Ghibli-style artwork using deep learning techniques with Stable Diffusion models. The README file format helps users understand how to set up and use your project effectively.

