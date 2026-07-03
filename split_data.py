import os
import random
import shutil

# Path to your images folder
images_path = 'C:/Users/14025/Desktop/BaceballTrak/images'

# Create train and val directories if they don't exist
train_path = os.path.join(images_path, 'train_images')
val_path = os.path.join(images_path, 'val_images')

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Get a list of all the image files
all_images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]  # or .png, etc.

# Shuffle the images randomly
random.shuffle(all_images)

# Split the images (80% for training, 20% for validation)
split_index = int(len(all_images) * 0.8)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Move the images to the appropriate folders
for img in train_images:
    shutil.move(os.path.join(images_path, img), os.path.join(train_path, img))

for img in val_images:
    shutil.move(os.path.join(images_path, img), os.path.join(val_path, img))

print(f"Moved {len(train_images)} images to {train_path}")
print(f"Moved {len(val_images)} images to {val_path}")
