import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

# --- 1. Configuration ---
num_classes = 23
img_rows, img_cols = 48, 48
batch_size = 32
# NOTE: Update these paths to your local directory structure
train_data_dir = 'E:/Desktop/biometric/archive (4)/train' 
validation_data_dir = 'E:/Desktop/biometric/archive (4)/test'
epochs = 25 # Set to 25 as requested

# --- 2. Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomTranslation(0.4, 0.4),
    tf.keras.layers.RandomFlip("horizontal")
])

# --- 3. Dataset Loading and Preprocessing ---
print("Loading Training Data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    color_mode='grayscale',
    image_size=(img_rows, img_cols),
    batch_size=batch_size,
    shuffle=True
)

print("Loading Validation Data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    image_size=(img_rows, img_cols),
    batch_size=batch_size,
    shuffle=True
)

# Normalize the data (Rescaling to [0, 1])
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Configure datasets for performance, normalization, and augmentation
AUTOTUNE = tf.data.AUTOTUNE

# Apply normalization, then augmentation to the training set
train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=AUTOTUNE
)
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# Apply only normalization to the validation set
validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# --- 4. Model Architecture (Sequential CNN) ---
model = Sequential()

# Block-1: Conv -> BN -> ELU -> Conv -> BN -> ELU -> MaxPool -> Dropout
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2: Conv -> BN -> ELU -> Conv -> BN -> ELU -> MaxPool -> Dropout
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3: Conv -> BN -> ELU -> Conv -> BN -> ELU -> MaxPool -> Dropout
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4: Conv -> BN -> ELU -> Conv -> BN -> ELU -> MaxPool -> Dropout
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5: Flatten -> Dense -> BN -> ELU -> Dropout
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6: Dense -> BN -> ELU -> Dropout
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7: Output Layer
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print("\n--- Model Summary ---")
print(model.summary())
print("---------------------\n")

# --- 5. Callbacks Setup ---
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'skin_disease.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3, # Reduces LR after 3 epochs with no improvement
    verbose=1,
    min_delta=0.0001
)

# Remove EarlyStopping to ensure 25 epochs
callbacks = [checkpoint, reduce_lr]

# --- 6. Model Compilation ---
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# --- 7. Model Training ---
print("Starting Training...")
history = model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_dataset
)
print("Training Complete.")

# --- 8. Model Saving ---
models_dir = 'models' 
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
final_model_path = os.path.join(models_dir, 'cnn_skin_disease_model.h5')
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# --- 9. Plot Training History ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()