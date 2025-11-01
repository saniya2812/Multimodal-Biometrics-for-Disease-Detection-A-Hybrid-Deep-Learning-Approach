# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
from datetime import datetime

# Colab persistence setup
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    # Mount Google Drive
    drive.mount('/content/drive')
    # Set the project directory in Google Drive
    PROJECT_DIR = '/content/drive/MyDrive/biometric_project'
else:
    PROJECT_DIR = os.getcwd()
    print(f"Working locally, project directory: {PROJECT_DIR}")

# If in Colab, create project directory
if IN_COLAB:
    os.makedirs(PROJECT_DIR, exist_ok=True)
    print(f"Working in Colab, project directory: {PROJECT_DIR}")

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import random
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Sequential, Model
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, 
                          Dropout, Input, GlobalAveragePooling2D, BatchNormalization)
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, 
                             accuracy_score, classification_report)
import cv2

# Set style for plots
plt.style.use('default')

# Define disease categories
our_folders = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Bullous Disease Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
]

# Update these paths according to your directory structure
root_dir = r'E:\Desktop\biometric\archive (4)\train'
test_dir = r'E:\Desktop\biometric\archive (4)\test'

# Helper to safely list files in a directory (returns filenames only)
def list_files_in_dir(path):
    try:
        if not os.path.exists(path):
            print(f"Warning: path does not exist: {path}")
            return []
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    except Exception as e:
        print(f"Warning: could not list files in {path}: {e}")
        return []

# Create path variables for all disease folders
acne_train_path = os.path.join(root_dir, 'Acne and Rosacea Photos')
actinic_train_path = os.path.join(root_dir, 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions')
atopic_train_path = os.path.join(root_dir, 'Atopic Dermatitis Photos')
bullous_train_path = os.path.join(root_dir, 'Bullous Disease Photos')
cellulitis_train_path = os.path.join(root_dir, 'Cellulitis Impetigo and other Bacterial Infections')
eczema_train_path = os.path.join(root_dir, 'Eczema Photos')
exanthems_train_path = os.path.join(root_dir, 'Exanthems and Drug Eruptions')
herpes_train_path = os.path.join(root_dir, 'Herpes HPV and other STDs Photos')
light_diseases_train_path = os.path.join(root_dir, 'Light Diseases and Disorders of Pigmentation')
lupus_train_path = os.path.join(root_dir, 'Lupus and other Connective Tissue diseases')
melanoma_train_path = os.path.join(root_dir, 'Melanoma Skin Cancer Nevi and Moles')
poison_ivy_train_path = os.path.join(root_dir, 'Poison Ivy Photos and other Contact Dermatitis')
scabies_train_path = os.path.join(root_dir, 'Scabies Lyme Disease and other Infestations and Bites')
seborrheic_train_path = os.path.join(root_dir, 'Seborrheic Keratoses and other Benign Tumors')
systemic_train_path = os.path.join(root_dir, 'Systemic Disease')
tinea_train_path = os.path.join(root_dir, 'Tinea Ringworm Candidiasis and other Fungal Infections')
urticaria_train_path = os.path.join(root_dir, 'Urticaria Hives')
vascular_tumors_train_path = os.path.join(root_dir, 'Vascular Tumors')
vasculitis_train_path = os.path.join(root_dir, 'Vasculitis Photos')

# Create test path variables for all disease folders
acne_test_path = os.path.join(test_dir, 'Acne and Rosacea Photos')
actinic_test_path = os.path.join(test_dir, 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions')
atopic_test_path = os.path.join(test_dir, 'Atopic Dermatitis Photos')
bullous_test_path = os.path.join(test_dir, 'Bullous Disease Photos')
cellulitis_test_path = os.path.join(test_dir, 'Cellulitis Impetigo and other Bacterial Infections')
eczema_test_path = os.path.join(test_dir, 'Eczema Photos')
exanthems_test_path = os.path.join(test_dir, 'Exanthems and Drug Eruptions')
herpes_test_path = os.path.join(test_dir, 'Herpes HPV and other STDs Photos')
light_diseases_test_path = os.path.join(test_dir, 'Light Diseases and Disorders of Pigmentation')
lupus_test_path = os.path.join(test_dir, 'Lupus and other Connective Tissue diseases')
melanoma_test_path = os.path.join(test_dir, 'Melanoma Skin Cancer Nevi and Moles')
poison_ivy_test_path = os.path.join(test_dir, 'Poison Ivy Photos and other Contact Dermatitis')
scabies_test_path = os.path.join(test_dir, 'Scabies Lyme Disease and other Infestations and Bites')
seborrheic_test_path = os.path.join(test_dir, 'Seborrheic Keratoses and other Benign Tumors')
systemic_test_path = os.path.join(test_dir, 'Systemic Disease')
tinea_test_path = os.path.join(test_dir, 'Tinea Ringworm Candidiasis and other Fungal Infections')
urticaria_test_path = os.path.join(test_dir, 'Urticaria Hives')
vascular_tumors_test_path = os.path.join(test_dir, 'Vascular Tumors')
vasculitis_test_path = os.path.join(test_dir, 'Vasculitis Photos')

# Get training files for all disease categories
print("Loading training files...")
acne_train_files = list_files_in_dir(acne_train_path)
actinic_train_files = list_files_in_dir(actinic_train_path)
atopic_train_files = list_files_in_dir(atopic_train_path)
bullous_train_files = list_files_in_dir(bullous_train_path)
cellulitis_train_files = list_files_in_dir(cellulitis_train_path)
eczema_train_files = list_files_in_dir(eczema_train_path)
exanthems_train_files = list_files_in_dir(exanthems_train_path)
herpes_train_files = list_files_in_dir(herpes_train_path)
light_diseases_train_files = list_files_in_dir(light_diseases_train_path)
lupus_train_files = list_files_in_dir(lupus_train_path)
melanoma_train_files = list_files_in_dir(melanoma_train_path)
poison_ivy_train_files = list_files_in_dir(poison_ivy_train_path)
scabies_train_files = list_files_in_dir(scabies_train_path)
seborrheic_train_files = list_files_in_dir(seborrheic_train_path)
systemic_train_files = list_files_in_dir(systemic_train_path)
tinea_train_files = list_files_in_dir(tinea_train_path)
urticaria_train_files = list_files_in_dir(urticaria_train_path)
vascular_tumors_train_files = list_files_in_dir(vascular_tumors_train_path)
vasculitis_train_files = list_files_in_dir(vasculitis_train_path)

# Get test files for all disease categories
print("Loading test files...")
acne_test_files = list_files_in_dir(acne_test_path)
actinic_test_files = list_files_in_dir(actinic_test_path)
atopic_test_files = list_files_in_dir(atopic_test_path)
bullous_test_files = list_files_in_dir(bullous_test_path)
cellulitis_test_files = list_files_in_dir(cellulitis_test_path)
eczema_test_files = list_files_in_dir(eczema_test_path)
exanthems_test_files = list_files_in_dir(exanthems_test_path)
herpes_test_files = list_files_in_dir(herpes_test_path)
light_diseases_test_files = list_files_in_dir(light_diseases_test_path)
lupus_test_files = list_files_in_dir(lupus_test_path)
melanoma_test_files = list_files_in_dir(melanoma_test_path)
poison_ivy_test_files = list_files_in_dir(poison_ivy_test_path)
scabies_test_files = list_files_in_dir(scabies_test_path)
seborrheic_test_files = list_files_in_dir(seborrheic_test_path)
systemic_test_files = list_files_in_dir(systemic_test_path)
tinea_test_files = list_files_in_dir(tinea_test_path)
urticaria_test_files = list_files_in_dir(urticaria_test_path)
vascular_tumors_test_files = list_files_in_dir(vascular_tumors_test_path)
vasculitis_test_files = list_files_in_dir(vasculitis_test_path)

# Create DataFrames for all disease categories
print("Creating training DataFrames...")
acne_df = pd.DataFrame()
acne_df['Image'] = [os.path.join(acne_train_path, img) for img in acne_train_files]
acne_df['Label'] = "acne"

actinic_df = pd.DataFrame()
actinic_df['Image'] = [os.path.join(actinic_train_path, img) for img in actinic_train_files]
actinic_df['Label'] = "actinic"

atopic_df = pd.DataFrame()
atopic_df['Image'] = [os.path.join(atopic_train_path, img) for img in atopic_train_files]
atopic_df['Label'] = "atopic"

bullous_df = pd.DataFrame()
bullous_df['Image'] = [os.path.join(bullous_train_path, img) for img in bullous_train_files]
bullous_df['Label'] = "bullous"

cellulitis_df = pd.DataFrame()
cellulitis_df['Image'] = [os.path.join(cellulitis_train_path, img) for img in cellulitis_train_files]
cellulitis_df['Label'] = "cellulitis"

eczema_df = pd.DataFrame()
eczema_df['Image'] = [os.path.join(eczema_train_path, img) for img in eczema_train_files]
eczema_df['Label'] = "eczema"

exanthems_df = pd.DataFrame()
exanthems_df['Image'] = [os.path.join(exanthems_train_path, img) for img in exanthems_train_files]
exanthems_df['Label'] = "exanthems"

herpes_df = pd.DataFrame()
herpes_df['Image'] = [os.path.join(herpes_train_path, img) for img in herpes_train_files]
herpes_df['Label'] = "herpes"

light_diseases_df = pd.DataFrame()
light_diseases_df['Image'] = [os.path.join(light_diseases_train_path, img) for img in light_diseases_train_files]
light_diseases_df['Label'] = "light_diseases"

lupus_df = pd.DataFrame()
lupus_df['Image'] = [os.path.join(lupus_train_path, img) for img in lupus_train_files]
lupus_df['Label'] = "lupus"

melanoma_df = pd.DataFrame()
melanoma_df['Image'] = [os.path.join(melanoma_train_path, img) for img in melanoma_train_files]
melanoma_df['Label'] = "melanoma"

poison_ivy_df = pd.DataFrame()
poison_ivy_df['Image'] = [os.path.join(poison_ivy_train_path, img) for img in poison_ivy_train_files]
poison_ivy_df['Label'] = "poison_ivy"

scabies_df = pd.DataFrame()
scabies_df['Image'] = [os.path.join(scabies_train_path, img) for img in scabies_train_files]
scabies_df['Label'] = "scabies"

seborrheic_df = pd.DataFrame()
seborrheic_df['Image'] = [os.path.join(seborrheic_train_path, img) for img in seborrheic_train_files]
seborrheic_df['Label'] = "seborrheic"

systemic_df = pd.DataFrame()
systemic_df['Image'] = [os.path.join(systemic_train_path, img) for img in systemic_train_files]
systemic_df['Label'] = "systemic"

tinea_df = pd.DataFrame()
tinea_df['Image'] = [os.path.join(tinea_train_path, img) for img in tinea_train_files]
tinea_df['Label'] = "tinea"

urticaria_df = pd.DataFrame()
urticaria_df['Image'] = [os.path.join(urticaria_train_path, img) for img in urticaria_train_files]
urticaria_df['Label'] = "urticaria"

vascular_tumors_df = pd.DataFrame()
vascular_tumors_df['Image'] = [os.path.join(vascular_tumors_train_path, img) for img in vascular_tumors_train_files]
vascular_tumors_df['Label'] = "vascular_tumors"

vasculitis_df = pd.DataFrame()
vasculitis_df['Image'] = [os.path.join(vasculitis_train_path, img) for img in vasculitis_train_files]
vasculitis_df['Label'] = "vasculitis"

# Combine all training DataFrames
final_df = pd.concat([
    acne_df, actinic_df, atopic_df, bullous_df, cellulitis_df,
    eczema_df, exanthems_df, herpes_df, light_diseases_df, lupus_df,
    melanoma_df, poison_ivy_df, scabies_df, seborrheic_df, systemic_df,
    tinea_df, urticaria_df, vascular_tumors_df, vasculitis_df
], ignore_index=True)

print(f"Final training DataFrame shape: {final_df.shape}")

# Create test DataFrames
print("Creating test DataFrames...")
acne_test_df = pd.DataFrame()
acne_test_df['Image'] = [os.path.join(acne_test_path, img) for img in acne_test_files]
acne_test_df['Label'] = "acne"

actinic_test_df = pd.DataFrame()
actinic_test_df['Image'] = [os.path.join(actinic_test_path, img) for img in actinic_test_files]
actinic_test_df['Label'] = "actinic"

atopic_test_df = pd.DataFrame()
atopic_test_df['Image'] = [os.path.join(atopic_test_path, img) for img in atopic_test_files]
atopic_test_df['Label'] = "atopic"

bullous_test_df = pd.DataFrame()
bullous_test_df['Image'] = [os.path.join(bullous_test_path, img) for img in bullous_test_files]
bullous_test_df['Label'] = "bullous"

cellulitis_test_df = pd.DataFrame()
cellulitis_test_df['Image'] = [os.path.join(cellulitis_test_path, img) for img in cellulitis_test_files]
cellulitis_test_df['Label'] = "cellulitis"

eczema_test_df = pd.DataFrame()
eczema_test_df['Image'] = [os.path.join(eczema_test_path, img) for img in eczema_test_files]
eczema_test_df['Label'] = "eczema"

exanthems_test_df = pd.DataFrame()
exanthems_test_df['Image'] = [os.path.join(exanthems_test_path, img) for img in exanthems_test_files]
exanthems_test_df['Label'] = "exanthems"

herpes_test_df = pd.DataFrame()
herpes_test_df['Image'] = [os.path.join(herpes_test_path, img) for img in herpes_test_files]
herpes_test_df['Label'] = "herpes"

light_diseases_test_df = pd.DataFrame()
light_diseases_test_df['Image'] = [os.path.join(light_diseases_test_path, img) for img in light_diseases_test_files]
light_diseases_test_df['Label'] = "light_diseases"

lupus_test_df = pd.DataFrame()
lupus_test_df['Image'] = [os.path.join(lupus_test_path, img) for img in lupus_test_files]
lupus_test_df['Label'] = "lupus"

melanoma_test_df = pd.DataFrame()
melanoma_test_df['Image'] = [os.path.join(melanoma_test_path, img) for img in melanoma_test_files]
melanoma_test_df['Label'] = "melanoma"

poison_ivy_test_df = pd.DataFrame()
poison_ivy_test_df['Image'] = [os.path.join(poison_ivy_test_path, img) for img in poison_ivy_test_files]
poison_ivy_test_df['Label'] = "poison_ivy"

scabies_test_df = pd.DataFrame()
scabies_test_df['Image'] = [os.path.join(scabies_test_path, img) for img in scabies_test_files]
scabies_test_df['Label'] = "scabies"

seborrheic_test_df = pd.DataFrame()
seborrheic_test_df['Image'] = [os.path.join(seborrheic_test_path, img) for img in seborrheic_test_files]
seborrheic_test_df['Label'] = "seborrheic"

systemic_test_df = pd.DataFrame()
systemic_test_df['Image'] = [os.path.join(systemic_test_path, img) for img in systemic_test_files]
systemic_test_df['Label'] = "systemic"

tinea_test_df = pd.DataFrame()
tinea_test_df['Image'] = [os.path.join(tinea_test_path, img) for img in tinea_test_files]
tinea_test_df['Label'] = "tinea"

urticaria_test_df = pd.DataFrame()
urticaria_test_df['Image'] = [os.path.join(urticaria_test_path, img) for img in urticaria_test_files]
urticaria_test_df['Label'] = "urticaria"

vascular_tumors_test_df = pd.DataFrame()
vascular_tumors_test_df['Image'] = [os.path.join(vascular_tumors_test_path, img) for img in vascular_tumors_test_files]
vascular_tumors_test_df['Label'] = "vascular_tumors"

vasculitis_test_df = pd.DataFrame()
vasculitis_test_df['Image'] = [os.path.join(vasculitis_test_path, img) for img in vasculitis_test_files]
vasculitis_test_df['Label'] = "vasculitis"

# Combine all test DataFrames
final_test_df = pd.concat([
    acne_test_df, actinic_test_df, atopic_test_df, bullous_test_df, cellulitis_test_df,
    eczema_test_df, exanthems_test_df, herpes_test_df, light_diseases_test_df, lupus_test_df,
    melanoma_test_df, poison_ivy_test_df, scabies_test_df, seborrheic_test_df, systemic_test_df,
    tinea_test_df, urticaria_test_df, vascular_tumors_test_df, vasculitis_test_df
], ignore_index=True)

print(f"Final test DataFrame shape: {final_test_df.shape}")

# Model Configuration
print("Setting up the model...")
image_size = (256, 256)
batch_size = 32

# Create datasets using tf.keras.utils.image_dataset_from_directory
print("Creating data generators...")
train_generator = tf.keras.utils.image_dataset_from_directory(
    root_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    interpolation='bilinear',
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training'
)

valid_generator = tf.keras.utils.image_dataset_from_directory(
    root_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    interpolation='bilinear',
    batch_size=batch_size,
    shuffle=False,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

# Create test generator
test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    interpolation='bilinear',
    batch_size=batch_size,
    shuffle=False
)

print(f"Number of training batches: {len(train_generator)}")
print(f"Number of validation batches: {len(valid_generator)}")
print(f"Number of test batches: {len(test_generator)}")

# Get number of classes and class names
num_classes = len(train_generator.class_names)
class_names = train_generator.class_names
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Build VGG16 Model
print("Building VGG16 model...")
res = VGG16(weights='imagenet', include_top=False,
            input_shape=(256, 256, 3))

# Setting the trainable to false
res.trainable = False

x = res.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(res.input, x)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["categorical_accuracy"])

print("Model summary:")
model.summary()

# Callbacks
custom_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    mode='min',
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# Train the model
print("Starting training...")
history = model.fit(
    train_generator, 
    epochs=25,  # Fixed to 25 epochs as requested
    validation_data=valid_generator,
    callbacks=[custom_early_stopping, checkpoint]
)

print("Training completed!")

# Create a models directory inside the project and save the trained models and metadata there
if IN_COLAB:
    models_dir = os.path.join(PROJECT_DIR, 'models')
else:
    models_dir = os.path.join(os.path.dirname(__file__), 'models') if '__file__' in globals() else os.path.join(os.getcwd(), 'models')
os.makedirs(models_dir, exist_ok=True)

# Save a timestamp with the model to track different training runs
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

try:
    # Save final model in Keras format (recommended)
    final_model_path = os.path.join(models_dir, 'final_model_19_vgg_diseases.keras')
    model.save(final_model_path)
    print(f"Saved final Keras model to: {final_model_path}")

    # Also save in HDF5 format for compatibility (optional)
    final_model_h5_path = os.path.join(models_dir, 'final_model_19_vgg_diseases.h5')
    model.save(final_model_h5_path)
    print(f"Saved HDF5 model to: {final_model_h5_path}")

    # Save class indices mapping so callers can map predictions back to labels
    import json
    class_indices_path = os.path.join(models_dir, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=4)
    print(f"Saved class indices to: {class_indices_path}")

except Exception as e:
    print(f"Warning: could not save model artifacts to {models_dir}: {e}")



# Plot training history
plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# =============================================================================
# TESTING AND PREDICTION SECTION
# =============================================================================

print("\n" + "="*60)
print("TESTING AND PREDICTION RESULTS")
print("="*60)

# Evaluate the model on test data
print("Evaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get true labels and predictions
test_true = test_generator.classes
test_pred_raw = model.predict(test_generator)
test_pred = np.argmax(test_pred_raw, axis=1)

# Calculate accuracy
vgg_acc = accuracy_score(test_true, test_pred)
print(f"\nVGG16 Model Overall Test Accuracy: {vgg_acc * 100:.2f}%")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(test_true, test_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(test_true, test_pred)
plt.figure(figsize=(15, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format='d')
plt.title('Confusion Matrix - Test Data', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# =============================================================================
# INDIVIDUAL IMAGE PREDICTION FUNCTION
# =============================================================================

def predict_image_class(image_path, true_value=None):
    """
    Predict the class of a single image and display results
    """
    try:
        # Load and preprocess image
        img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(score)
        predicted_class = class_names[predicted_class_idx]
        confidence = 100 * np.max(score)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(score)[-3:][::-1]
        top_3_classes = [class_names[i] for i in top_3_indices]
        top_3_confidences = [100 * score[i] for i in top_3_indices]
        
        # Print results
        print(f"\n📊 PREDICTION RESULTS for: {os.path.basename(image_path)}")
        print("-" * 50)
        print(f"🖼️  Image: {os.path.basename(image_path)}")
        
        if true_value:
            print(f"✅ True Label: {true_value}")
        
        print(f"🎯 Predicted Class: {predicted_class}")
        print(f"📈 Confidence: {confidence:.2f}%")
        
        print("\n🏆 Top 3 Predictions:")
        for i, (cls, conf) in enumerate(zip(top_3_classes, top_3_confidences)):
            print(f"   {i+1}. {cls}: {conf:.2f}%")
        
        # Display the image with predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        ax1.imshow(img / 255.)
        ax1.axis('off')
        
        # Set title based on correctness
        if true_value:
            title_color = 'green' if true_value.lower() == predicted_class.lower() else 'red'
            correctness = "✓ CORRECT" if true_value.lower() == predicted_class.lower() else "✗ INCORRECT"
            ax1.set_title(f'{correctness}\nTrue: {true_value}', color=title_color, fontsize=14, pad=20)
        else:
            ax1.set_title(f'Predicted: {predicted_class}', fontsize=14, pad=20)
        
        # Create confidence bar chart
        y_pos = np.arange(len(top_3_classes))
        ax2.barh(y_pos, top_3_confidences, color=['#2E8B57', '#FFA500', '#FF6347'])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_3_classes)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('Top 3 Predictions')
        ax2.set_xlim(0, 100)
        
        # Add confidence values on bars
        for i, v in enumerate(top_3_confidences):
            ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence, top_3_classes, top_3_confidences
        
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None, None, None, None

# =============================================================================
# TEST MULTIPLE RANDOM IMAGES FROM TEST SET
# =============================================================================

def test_random_images(num_images=10):
    """
    Test random images from the test set and show predictions
    """
    print(f"\n🧪 TESTING {num_images} RANDOM IMAGES FROM TEST SET")
    print("=" * 60)
    
    # Get random samples from test set
    if len(final_test_df) > 0:
        test_samples = final_test_df.sample(n=min(num_images, len(final_test_df)), random_state=42)
        
        correct_predictions = 0
        total_predictions = len(test_samples)
        
        for idx, (_, row) in enumerate(test_samples.iterrows()):
            image_path = row['Image']
            true_label = row['Label']
            
            print(f"\n{'='*50}")
            print(f"📸 TEST IMAGE {idx + 1}/{total_predictions}")
            print(f"{'='*50}")
            
            if os.path.exists(image_path):
                predicted_class, confidence, _, _ = predict_image_class(image_path, true_label)
                
                if predicted_class and predicted_class.lower() == true_label.lower():
                    correct_predictions += 1
            else:
                print(f"❌ Image not found: {image_path}")
        
        # Print overall testing summary
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n🎯 RANDOM TESTING SUMMARY:")
        print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.2f}%")
    else:
        print("❌ No test images found in the test DataFrame")

# =============================================================================
# PREDICT ON SPECIFIC TEST IMAGES
# =============================================================================

def predict_specific_test_images():
    """
    Predict on specific test images from each class
    """
    print(f"\n🎯 PREDICTING SPECIFIC TEST IMAGES FROM EACH CLASS")
    print("=" * 60)
    
    # Get one image from each class in test set
    class_samples = {}
    for class_name in class_names:
        class_images = final_test_df[final_test_df['Label'] == class_name]
        if len(class_images) > 0:
            class_samples[class_name] = class_images.iloc[0]
    
    for class_name, sample in class_samples.items():
        image_path = sample['Image']
        true_label = sample['Label']
        
        print(f"\n🔍 Testing {class_name} class:")
        print("-" * 30)
        
        if os.path.exists(image_path):
            predict_image_class(image_path, true_label)
        else:
            print(f"❌ Image not found: {image_path}")

# =============================================================================
# RUN PREDICTIONS
# =============================================================================

# Test random images
test_random_images(num_images=5)

# Test specific images from each class
predict_specific_test_images()

# =============================================================================
# BATCH PREDICTION ON ENTIRE TEST SET
# =============================================================================

print(f"\n📊 BATCH PREDICTION ON ENTIRE TEST SET")
print("=" * 50)

# Get all predictions
all_predictions = model.predict(test_generator)
all_predicted_classes = np.argmax(all_predictions, axis=1)
all_true_classes = test_generator.classes

# Calculate per-class accuracy
class_accuracy = {}
for i, class_name in enumerate(class_names):
    class_mask = all_true_classes == i
    if np.sum(class_mask) > 0:
        class_acc = np.mean(all_predicted_classes[class_mask] == i)
        class_accuracy[class_name] = class_acc * 100

# Display per-class accuracy
print("\n📈 PER-CLASS ACCURACY:")
print("-" * 30)
for class_name, acc in class_accuracy.items():
    print(f"   {class_name:.<30} {acc:5.1f}%")

# Overall statistics
print(f"\n📊 OVERALL TEST STATISTICS:")
print(f"   Total test images: {len(all_true_classes)}")
print(f"   Correct predictions: {np.sum(all_predicted_classes == all_true_classes)}")
print(f"   Overall accuracy: {vgg_acc * 100:.2f}%")

print("\n✅ Testing and prediction completed successfully!")