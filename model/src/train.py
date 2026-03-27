import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Stabilized CNN Architecture with Batch Normalization
def build_model(input_shape=(128, 128, 3), num_classes=7):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(), # <-- STABILIZER ADDED
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2), 
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(), # <-- STABILIZER ADDED
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(), # <-- STABILIZER ADDED
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(), # <-- STABILIZER ADDED
        layers.Activation('relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 32

    print("Loading Cleaned Training Data...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        "../data/HAM10000_cleaned/train", 
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    print("Loading Cleaned Validation Data...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        "../data/HAM10000_cleaned/val",
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Build and Compile (Using Standard CrossEntropy this time)
    model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=7)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy', # Reverting to standard, stable loss
        metrics=['accuracy']
    )
    
    # 3. Callbacks
    checkpoint_dir = "../checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
    )
    
    # NEW: Automatically reduces learning rate when stuck
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
    )

    print("\n🚀 Starting Stabilized Model Training...\n")
    
    # Keeping the class weights so it doesn't ignore the rare cancers
    disease_weights = {
        0: 4.37, 1: 2.78, 2: 1.30, 3: 12.44, 4: 1.28, 5: 0.21, 6: 10.04
    }
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30, 
        callbacks=[model_checkpoint, early_stopping, lr_scheduler],
        class_weight=disease_weights 
    )
    
    print(f"\n✅ Training Complete! Your best model is saved at: {checkpoint_path}")