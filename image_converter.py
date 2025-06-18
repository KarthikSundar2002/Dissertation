from PIL import Image
import os
from os.path import join

def convert_to_grayscale(input_dir, output_dir, size=256):
    """
    Convert all images in the input directory to grayscale and save them to the output directory.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save grayscale images
        size (int): Size to resize images to (default: 256)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                # Open image
                with Image.open(input_path) as image:
                    # Convert to grayscale
                    gray_image = image.convert('L')
                    # Resize image
                    gray_image = gray_image.resize((size, size), Image.Resampling.LANCZOS)
                    
                    # Save the grayscale image as PNG
                    image_output_path = join(output_dir, f"gray_{base_name}.png")
                    gray_image.save(image_output_path)
                    
                    print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_directory = "data"
    output_directory = "output/grayscale_images"
    convert_to_grayscale(input_directory, output_directory) 