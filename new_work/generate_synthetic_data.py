#!/usr/bin/env python3
"""
Generate synthetic ocular disease data using Stable Diffusion
This script creates synthetic images for the missing t-SNE dataset
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

class SyntheticDataGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "runwayml/stable-diffusion-v1-5"
        
        # Output directories
        self.synthetic_dir = "/kaggle/working/synthetic_data/"
        
        # Generation parameters
        self.num_images_per_class = 200
        self.image_size = 512
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        
        # Disease classes and prompts
        self.classes = {
            "G": "Glaucoma",
            "C": "Cataract", 
            "A": "Age Related Macular Degeneration",
            "H": "Hypertension",
            "M": "Myopia"
        }
        
        # Carefully crafted prompts for each disease
        self.prompts = {
            "G": "fundus photograph, glaucoma, optic nerve head cupping, enlarged cup-to-disc ratio, retinal nerve fiber layer defects, medical imaging, high resolution, clinical photography",
            "C": "fundus photograph, cataract, lens opacity, cloudy lens, lens changes, medical imaging, high resolution, clinical photography",
            "A": "fundus photograph, age-related macular degeneration, drusen deposits, retinal pigment epithelium changes, macular changes, medical imaging, high resolution, clinical photography",
            "H": "fundus photograph, hypertensive retinopathy, arteriolar narrowing, cotton wool spots, flame hemorrhages, medical imaging, high resolution, clinical photography",
            "M": "fundus photograph, myopic retinopathy, elongated eye, peripapillary atrophy, myopic crescent, medical imaging, high resolution, clinical photography"
        }
        
        print(f"Using device: {self.device}")
        print(f"Output directory: {self.synthetic_dir}")
        
        # Create output directories
        os.makedirs(self.synthetic_dir, exist_ok=True)
        for class_name in self.classes.values():
            class_dir = os.path.join(self.synthetic_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    def load_model(self):
        """Load Stable Diffusion pipeline"""
        print("Loading Stable Diffusion model...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)
        print(f"Model loaded successfully on {self.device}")
    
    def generate_images(self):
        """Generate synthetic images for each disease class"""
        synthetic_metadata = []
        
        for class_short, class_name in self.classes.items():
            print(f"\n{'='*60}")
            print(f"Generating images for {class_name} ({class_short})")
            print(f"{'='*60}")
            
            prompt = self.prompts[class_short]
            class_dir = os.path.join(self.synthetic_dir, class_name)
            
            print(f"Prompt: {prompt}")
            print(f"Generating {self.num_images_per_class} images...")
            
            # Generate images in batches to manage memory
            batch_size = 4
            num_batches = (self.num_images_per_class + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.num_images_per_class)
                current_batch_size = end_idx - start_idx
                
                print(f"  Batch {batch_idx + 1}/{num_batches} (images {start_idx + 1}-{end_idx})")
                
                # Generate batch of images
                images = self.pipe(
                    [prompt] * current_batch_size,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    height=self.image_size,
                    width=self.image_size
                ).images
                
                # Save images
                for i, image in enumerate(images):
                    img_idx = start_idx + i
                    filename = f"{class_short}_{img_idx:04d}.png"
                    image_path = os.path.join(class_dir, filename)
                    image.save(image_path)
                    
                    # Add to metadata
                    synthetic_metadata.append({
                        'filename': filename,
                        'class': class_short,
                        'labels': f"['{class_short}']",
                        'source': 'synthetic',
                        'prompt': prompt,
                        'disease_name': class_name
                    })
            
            print(f"✅ Completed {class_name}: {self.num_images_per_class} images generated")
        
        return synthetic_metadata
    
    def save_metadata(self, synthetic_metadata):
        """Save metadata CSV"""
        synthetic_df = pd.DataFrame(synthetic_metadata)
        metadata_path = '/kaggle/working/synthetic_metadata.csv'
        synthetic_df.to_csv(metadata_path, index=False)
        
        print(f"\n📊 Synthetic Data Summary:")
        print(f"Total images: {len(synthetic_df)}")
        print("\nPer class distribution:")
        print(synthetic_df['disease_name'].value_counts())
        
        print(f"\n💾 Metadata saved to: {metadata_path}")
        print(f"📁 Images saved to: {self.synthetic_dir}")
        
        return synthetic_df

def main():
    """Main function to generate synthetic data"""
    print("🚀 Starting synthetic ocular disease data generation...")
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Load model
    generator.load_model()
    
    # Generate images
    synthetic_metadata = generator.generate_images()
    
    # Save metadata
    synthetic_df = generator.save_metadata(synthetic_metadata)
    
    print("\n🎉 Synthetic data generation completed!")
    print(f"Generated {len(synthetic_metadata)} synthetic images")
    
    return synthetic_df

if __name__ == "__main__":
    synthetic_df = main()
