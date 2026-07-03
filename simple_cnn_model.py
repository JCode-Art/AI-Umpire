import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
import os

# Define dataset directories
train_dir = 'C:/Users/14025/Desktop/BaceballTrak/images/train_images'
val_dir = 'C:/Users/14025/Desktop/BaceballTrak/images/val_images'

# Ensure dataset directories exist
assert os.path.exists(train_dir) and os.path.exists(val_dir), "One or more dataset directories not found."
assert len(os.listdir(train_dir)) > 0 and len(os.listdir(val_dir)) > 0, "Dataset directories are empty."

print("Train images:", os.listdir(train_dir))
print("Validation images:", os.listdir(val_dir))

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load images with larger target size and batch size
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Larger image size
    batch_size=64,           # Increased batch size
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),  # Larger image size
    batch_size=64,           # Increased batch size
    class_mode='categorical'
)

print("Class indices:", train_data.class_indices)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),  # Adjust input shape
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)  # Increased patience
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


# Train model
history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=[reduce_lr, early_stop]
)

# Save trained model
model_save_dir = 'models'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'baseball_model.keras')
model.save(model_save_path)
print(f"Model saved successfully at {model_save_path}!")
