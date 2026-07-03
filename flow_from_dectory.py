import os

train_image_dir = 'C:/Users/14025/Desktop/BaceballTrak/images/train_images'
val_image_dir = 'C:/Users/14025/Desktop/BaceballTrak/images/val_images'

# Check contents of the directories
print("Training images in directory:")
for root, dirs, files in os.walk(train_image_dir):
    print(f'In {root}: {len(files)} files')

print("\nValidation images in directory:")
for root, dirs, files in os.walk(val_image_dir):
    print(f'In {root}: {len(files)} files')
