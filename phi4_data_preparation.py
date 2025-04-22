import os
import json
import argparse
import random
from PIL import Image
from tqdm import tqdm
import shutil
from pathlib import Path

# Categories from the moderation system
CATEGORIES = [
    "PORNOGRAPHY", "GORE", "WEAPONS", "EXTREMISM", "DRUGS", 
    "GAMBLING", "ID_CARDS", "DOCUMENTS", "FACE", 
    "SWIM_UNDERWEAR", "ALCOHOL", "Safe Content"
]

# Mapping of your dataset folders to moderation categories
FOLDER_TO_CATEGORY = {
    "alcohol": "ALCOHOL",
    "credit_card": "ID_CARDS",
    "drug-name": "DRUGS", 
    "gore": "GORE",
    "hate": "EXTREMISM",
    "memes": "Safe Content",  # Assuming memes are generally safe
    "middle_finger": "Safe Content",  # Could be classified differently if needed
    "porn": "PORNOGRAPHY",
    "safe": "Safe Content",
    "self_harm": "GORE",
    "selfie": "FACE",
    "violence": "GORE",
    "weapons": "WEAPONS"
}

# Folders to skip (duplicates or other unwanted folders)
SKIP_FOLDERS = ["credit_card_images"]

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets for Phi-4 image classification")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing categorized images")
    parser.add_argument("--output_dir", type=str, default="./phi4_datasets", help="Output directory for prepared datasets")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data (rest will be validation)")
    parser.add_argument("--include_multi_label", action="store_true", help="Generate multi-label examples")
    parser.add_argument("--multi_label_percentage", type=float, default=0.2, help="Percentage of multi-label examples")
    return parser.parse_args()

def verify_image(image_path):
    """Verify if the image is valid and can be opened"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def create_dataset_json(images_by_category, output_path, multi_label=False, multi_label_pct=0.2):
    """Create a JSON dataset file from the categorized images"""
    dataset = []
    
    # First create single-label examples
    single_label_examples = []
    for category, images in images_by_category.items():
        for image_path in images:
            single_label_examples.append({
                "image_path": image_path,
                "labels": category
            })
    
    # Determine how many should be multi-label if requested
    dataset.extend(single_label_examples)
    
    # Create multi-label examples if requested
    if multi_label and len(CATEGORIES) > 1:
        num_multi = int(len(single_label_examples) * multi_label_pct / (1 - multi_label_pct))
        
        multi_label_examples = []
        for _ in range(num_multi):
            # Randomly select 2-3 categories
            num_categories = random.randint(2, min(3, len(CATEGORIES)))
            selected_categories = random.sample(CATEGORIES, num_categories)
            
            # For each selected category, pick a random image
            selected_images = []
            for category in selected_categories:
                if category in images_by_category and images_by_category[category]:
                    selected_images.append(random.choice(images_by_category[category]))
            
            if selected_images:
                # Use the first image but assign multiple labels
                multi_label_examples.append({
                    "image_path": selected_images[0],
                    "labels": ", ".join(selected_categories)
                })
        
        dataset.extend(multi_label_examples)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return len(dataset)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store images by category
    images_by_category = {category: [] for category in CATEGORIES}
    
    # Find all images in the input directory structure
    print("Scanning for images...")
    
    # Process folders based on the mapping
    for folder, category in FOLDER_TO_CATEGORY.items():
        folder_path = os.path.join(args.input_dir, folder)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Processing folder: {folder} → {category}")
            
            for root, _, files in os.walk(folder_path):
                for file in tqdm(files, desc=f"Scanning {folder}", leave=False):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                        image_path = os.path.join(root, file)
                        if verify_image(image_path):
                            images_by_category[category].append(image_path)
    
    # Check for any folders in the input directory that aren't in our mapping
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path) and item not in FOLDER_TO_CATEGORY and item not in SKIP_FOLDERS:
            print(f"Warning: Folder '{item}' found but not mapped to any category. Skipping.")

    
    # Report on found images
    total_images = sum(len(images) for images in images_by_category.values())
    print(f"Found {total_images} valid images across {len(CATEGORIES)} categories")
    for category, images in images_by_category.items():
        print(f"  {category}: {len(images)} images")
    
    # Split into train and validation sets
    train_images = {category: [] for category in CATEGORIES}
    val_images = {category: [] for category in CATEGORIES}
    
    for category, images in images_by_category.items():
        random.shuffle(images)
        split_idx = int(len(images) * args.train_ratio)
        train_images[category] = images[:split_idx]
        val_images[category] = images[split_idx:]
    
    # Create train and validation JSON files
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    
    train_count = create_dataset_json(
        train_images, 
        train_path, 
        multi_label=args.include_multi_label, 
        multi_label_pct=args.multi_label_percentage
    )
    
    val_count = create_dataset_json(
        val_images, 
        val_path, 
        multi_label=args.include_multi_label, 
        multi_label_pct=args.multi_label_percentage
    )
    
    print(f"Created training dataset with {train_count} examples: {train_path}")
    print(f"Created validation dataset with {val_count} examples: {val_path}")
    
    # Create a small sample dataset for quick testing
    sample_images = {category: images[:min(5, len(images))] for category, images in train_images.items()}
    sample_path = os.path.join(args.output_dir, "sample.json")
    sample_count = create_dataset_json(
        sample_images,
        sample_path,
        multi_label=args.include_multi_label,
        multi_label_pct=args.multi_label_percentage
    )
    print(f"Created sample dataset with {sample_count} examples: {sample_path}")

if __name__ == "__main__":
    main()
