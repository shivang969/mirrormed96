import os
import cv2
from preprocessor import remove_hair

def process_dataset(input_base_dir, output_base_dir):
    # Create the base output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    splits = ['train', 'val']
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    total_images = 10015
    processed_count = 0

    print(f"Starting batch hair removal. Saving cleaned images to: {output_base_dir}\n")

    for split in splits:
        for cls in classes:
            input_dir = os.path.join(input_base_dir, split, cls)
            output_dir = os.path.join(output_base_dir, split, cls)
            
            # Create the exact same subfolder structure in the new directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            if not os.path.exists(input_dir):
                continue
                
            for img_name in os.listdir(input_dir):
                # Skip hidden files like .DS_Store on Mac
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                input_path = os.path.join(input_dir, img_name)
                output_path = os.path.join(output_dir, img_name)
                
                try:
                    # 1. Run your custom OpenCV algorithm
                    cleaned_img = remove_hair(input_path)
                    
                    # 2. Convert back to BGR for saving via OpenCV
                    cleaned_img_bgr = cv2.cvtColor(cleaned_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, cleaned_img_bgr)
                    
                    processed_count += 1
                    
                    # Print progress every 500 images so you know it hasn't crashed
                    if processed_count % 500 == 0:
                        print(f"[{processed_count}/{total_images}] images cleaned...")
                        
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

    print("\n✅ Batch processing complete! All images have been digitally shaved.")

if __name__ == "__main__":
    input_folder = "/Users/shivang/Desktop/tensorflow/myenv/mirrormed/model/data/HAM10000_split"
    output_folder = "../data/HAM10000_cleaned"
    
    process_dataset(input_folder, output_folder)