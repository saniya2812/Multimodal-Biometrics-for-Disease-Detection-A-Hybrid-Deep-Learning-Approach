
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sys


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

# Import data science libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import random

from PIL import Image
import cv2

# Import everything from tensorflow.keras to avoid conflicts
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten,
                                     Dropout, Input, GlobalAveragePooling2D,
                                     BatchNormalization)
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             accuracy_score, classification_report)

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
train_dirs = []
for i in our_folders:
    for folder_,_, files_ in os.walk(os.path.join(root_dir, i)):
        print(folder_)
        train_dirs.append(folder_)

# Get test files for all disease categories
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
def plotGridImages(d_name, list_files, train_path,nrows= 1, ncols=5):
    # for folder_name in our_folders:
    fig = plt.figure(1, figsize=(30, 30))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.05)
    print(f"{d_name}")
    loaded_images_count = 0
    for img_id in random.sample(list_files, min(ncols, len(list_files))):
        if loaded_images_count >= ncols:
            break
        ax = grid[loaded_images_count]
        image_dir_path = os.path.join(train_path, img_id)
        try:
            img = image.load_img(image_dir_path, target_size=(224, 224))
            img = image.img_to_array(img)
            ax.imshow(img / 255.)
            ax.text(10, 200, 'LABEL: %s' % d_name, color='k', backgroundcolor='w',\
            alpha=0.8)
            ax.axis('off')
            loaded_images_count += 1
        except Exception as e:
            print(f"Could not load image {image_dir_path}: {e}")
            # Skip this image and try to load another one if available
            continue

    # plt.tight_layout()
    plt.show()
    # Get training files for all disease categories
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
def plotGridImages(d_name, list_files, train_path,nrows= 1, ncols=5):
    fig = plt.figure(1, figsize=(30, 30))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.05)
    print(f"{d_name}")
    loaded_images_count = 0
    for img_id in random.sample(list_files, min(ncols, len(list_files))):
        if loaded_images_count >= ncols:
            break
        ax = grid[loaded_images_count]
        image_dir_path = os.path.join(train_path, img_id)
        try:
            img = image.load_img(image_dir_path, target_size=(224, 224), color_mode='rgb')
            img = image.img_to_array(img)
            ax.imshow(img / 255.)
            ax.text(10, 200, 'LABEL: %s' % d_name, color='k', backgroundcolor='w',\
            alpha=0.8)
            ax.axis('off')
            loaded_images_count += 1
        except Exception as e:
            print(f"Could not load image {image_dir_path}: {e}")
            # Skip this image and try to load another one if available
            continue

    plt.show()
    # Plot grid images for all 19 disease categories
plotGridImages('Acne and Rosacea', acne_train_files, acne_train_path, ncols=5)
plotGridImages('Actinic Keratosis', actinic_train_files, actinic_train_path, ncols=5)
plotGridImages('Atopic Dermatitis', atopic_train_files, atopic_train_path, ncols=5)
plotGridImages('Bullous Disease', bullous_train_files, bullous_train_path, ncols=5)
plotGridImages('Cellulitis', cellulitis_train_files, cellulitis_train_path, ncols=5)
plotGridImages('Eczema', eczema_train_files, eczema_train_path, ncols=5)
plotGridImages('Exanthems', exanthems_train_files, exanthems_train_path, ncols=5)
plotGridImages('Herpes HPV', herpes_train_files, herpes_train_path, ncols=5)
plotGridImages('Light Diseases', light_diseases_train_files, light_diseases_train_path, ncols=5)
plotGridImages('Lupus', lupus_train_files, lupus_train_path, ncols=5)
plotGridImages('Melanoma', melanoma_train_files, melanoma_train_path, ncols=5)
plotGridImages('Poison Ivy', poison_ivy_train_files, poison_ivy_train_path, ncols=5)
plotGridImages('Scabies Lyme', scabies_train_files, scabies_train_path, ncols=5)
plotGridImages('Seborrheic Keratoses', seborrheic_train_files, seborrheic_train_path, ncols=5)
plotGridImages('Systemic Disease', systemic_train_files, systemic_train_path, ncols=5)
plotGridImages('Tinea Ringworm', tinea_train_files, tinea_train_path, ncols=5)
plotGridImages('Urticaria Hives', urticaria_train_files, urticaria_train_path, ncols=5)
plotGridImages('Vascular Tumors', vascular_tumors_train_files, vascular_tumors_train_path, ncols=5)
plotGridImages('Vasculitis', vasculitis_train_files, vasculitis_train_path, ncols=5)
# Create DataFrames for all 19 disease categories
acne_df = pd.DataFrame()
acne_df['Image'] = [acne_train_path + '/' + img for img in acne_train_files]
acne_df['Label'] = "acne"
print(f"Acne shape: {acne_df.shape}")

actinic_df = pd.DataFrame()
actinic_df['Image'] = [actinic_train_path + '/' + img for img in actinic_train_files]
actinic_df['Label'] = "actinic"
print(f"Actinic shape: {actinic_df.shape}")

atopic_df = pd.DataFrame()
atopic_df['Image'] = [atopic_train_path + '/' + img for img in atopic_train_files]
atopic_df['Label'] = "atopic"
print(f"Atopic shape: {atopic_df.shape}")

bullous_df = pd.DataFrame()
bullous_df['Image'] = [bullous_train_path + '/' + img for img in bullous_train_files]
bullous_df['Label'] = "bullous"
print(f"Bullous shape: {bullous_df.shape}")

cellulitis_df = pd.DataFrame()
cellulitis_df['Image'] = [cellulitis_train_path + '/' + img for img in cellulitis_train_files]
cellulitis_df['Label'] = "cellulitis"
print(f"Cellulitis shape: {cellulitis_df.shape}")

eczema_df = pd.DataFrame()
eczema_df['Image'] = [eczema_train_path + '/' + img for img in eczema_train_files]
eczema_df['Label'] = "eczema"
print(f"Eczema shape: {eczema_df.shape}")

exanthems_df = pd.DataFrame()
exanthems_df['Image'] = [exanthems_train_path + '/' + img for img in exanthems_train_files]
exanthems_df['Label'] = "exanthems"
print(f"Exanthems shape: {exanthems_df.shape}")

herpes_df = pd.DataFrame()
herpes_df['Image'] = [herpes_train_path + '/' + img for img in herpes_train_files]
herpes_df['Label'] = "herpes"
print(f"Herpes shape: {herpes_df.shape}")

light_diseases_df = pd.DataFrame()
light_diseases_df['Image'] = [light_diseases_train_path + '/' + img for img in light_diseases_train_files]
light_diseases_df['Label'] = "light_diseases"
print(f"Light Diseases shape: {light_diseases_df.shape}")

lupus_df = pd.DataFrame()
lupus_df['Image'] = [lupus_train_path + '/' + img for img in lupus_train_files]
lupus_df['Label'] = "lupus"
print(f"Lupus shape: {lupus_df.shape}")

melanoma_df = pd.DataFrame()
melanoma_df['Image'] = [melanoma_train_path + '/' + img for img in melanoma_train_files]
melanoma_df['Label'] = "melanoma"
print(f"Melanoma shape: {melanoma_df.shape}")

poison_ivy_df = pd.DataFrame()
poison_ivy_df['Image'] = [poison_ivy_train_path + '/' + img for img in poison_ivy_train_files]
poison_ivy_df['Label'] = "poison_ivy"
print(f"Poison Ivy shape: {poison_ivy_df.shape}")

scabies_df = pd.DataFrame()
scabies_df['Image'] = [scabies_train_path + '/' + img for img in scabies_train_files]
scabies_df['Label'] = "scabies"
print(f"Scabies shape: {scabies_df.shape}")

seborrheic_df = pd.DataFrame()
seborrheic_df['Image'] = [seborrheic_train_path + '/' + img for img in seborrheic_train_files]
seborrheic_df['Label'] = "seborrheic"
print(f"Seborrheic shape: {seborrheic_df.shape}")

systemic_df = pd.DataFrame()
systemic_df['Image'] = [systemic_train_path + '/' + img for img in systemic_train_files]
systemic_df['Label'] = "systemic"
print(f"Systemic shape: {systemic_df.shape}")

tinea_df = pd.DataFrame()
tinea_df['Image'] = [tinea_train_path + '/' + img for img in tinea_train_files]
tinea_df['Label'] = "tinea"
print(f"Tinea shape: {tinea_df.shape}")

urticaria_df = pd.DataFrame()
urticaria_df['Image'] = [urticaria_train_path + '/' + img for img in urticaria_train_files]
urticaria_df['Label'] = "urticaria"
print(f"Urticaria shape: {urticaria_df.shape}")

vascular_tumors_df = pd.DataFrame()
vascular_tumors_df['Image'] = [vascular_tumors_train_path + '/' + img for img in vascular_tumors_train_files]
vascular_tumors_df['Label'] = "vascular_tumors"
print(f"Vascular Tumors shape: {vascular_tumors_df.shape}")

vasculitis_df = pd.DataFrame()
vasculitis_df['Image'] = [vasculitis_train_path + '/' + img for img in vasculitis_train_files]
vasculitis_df['Label'] = "vasculitis"
print(f"Vasculitis shape: {vasculitis_df.shape}")
# Append all 19 disease DataFrames to final_df
final_df = pd.concat([
    acne_df,
    actinic_df,
    atopic_df,
    bullous_df,
    cellulitis_df,
    eczema_df,
    exanthems_df,
    herpes_df,
    light_diseases_df,
    lupus_df,
    melanoma_df,
    poison_ivy_df,
    scabies_df,
    seborrheic_df,
    systemic_df,
    tinea_df,
    urticaria_df,
    vascular_tumors_df,
    vasculitis_df
], ignore_index=True)

print(f"Final DataFrame shape: {final_df.shape}")
final_df.shape
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define the figure size using plt.subplots()
# Example size: 10 inches wide by 6 inches tall
fig, ax = plt.subplots(figsize=(29, 8))

# 2. Pass the 'ax' object to the seaborn countplot
ax = sns.countplot(x=final_df['Label'],
                   order=final_df['Label'].value_counts(ascending=False).index,

                   ax=ax) # <-- Critical addition: ax=ax

# 3. Rest of your labeling code (which remains the same)
abs_values = final_df['Label'].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=abs_values)

# 4. Display the plot
plt.show()
final_test_df = pd.DataFrame()

################# Acne ##########
acne_test_df = pd.DataFrame()
acne_test_df['Image'] = [acne_test_path + '/' + img for img in acne_test_files]
acne_test_df['Label'] = "acne"

################# Actinic ##########
actinic_test_df = pd.DataFrame()
actinic_test_df['Image'] = [actinic_test_path + '/' + img for img in actinic_test_files]
actinic_test_df['Label'] = "actinic"

################# Atopic ##########
atopic_test_df = pd.DataFrame()
atopic_test_df['Image'] = [atopic_test_path + '/' + img for img in atopic_test_files]
atopic_test_df['Label'] = "atopic"

################# Bullous ##########
bullous_test_df = pd.DataFrame()
bullous_test_df['Image'] = [bullous_test_path + '/' + img for img in bullous_test_files]
bullous_test_df['Label'] = "bullous"

################# Cellulitis ##########
cellulitis_test_df = pd.DataFrame()
cellulitis_test_df['Image'] = [cellulitis_test_path + '/' + img for img in cellulitis_test_files]
cellulitis_test_df['Label'] = "cellulitis"

################# Eczema ##########
eczema_test_df = pd.DataFrame()
eczema_test_df['Image'] = [eczema_test_path + '/' + img for img in eczema_test_files]
eczema_test_df['Label'] = "eczema"

################# Exanthems ##########
exanthems_test_df = pd.DataFrame()
exanthems_test_df['Image'] = [exanthems_test_path + '/' + img for img in exanthems_test_files]
exanthems_test_df['Label'] = "exanthems"

################# Herpes ##########
herpes_test_df = pd.DataFrame()
herpes_test_df['Image'] = [herpes_test_path + '/' + img for img in herpes_test_files]
herpes_test_df['Label'] = "herpes"

################# Light Diseases ##########
light_diseases_test_df = pd.DataFrame()
light_diseases_test_df['Image'] = [light_diseases_test_path + '/' + img for img in light_diseases_test_files]
light_diseases_test_df['Label'] = "light_diseases"

################# Lupus ##########
lupus_test_df = pd.DataFrame()
lupus_test_df['Image'] = [lupus_test_path + '/' + img for img in lupus_test_files]
lupus_test_df['Label'] = "lupus"

################# Melanoma ##########
melanoma_test_df = pd.DataFrame()
melanoma_test_df['Image'] = [melanoma_test_path + '/' + img for img in melanoma_test_files]
melanoma_test_df['Label'] = "melanoma"

################# Poison Ivy ##########
poison_ivy_test_df = pd.DataFrame()
poison_ivy_test_df['Image'] = [poison_ivy_test_path + '/' + img for img in poison_ivy_test_files]
poison_ivy_test_df['Label'] = "poison_ivy"

################# Scabies ##########
scabies_test_df = pd.DataFrame()
scabies_test_df['Image'] = [scabies_test_path + '/' + img for img in scabies_test_files]
scabies_test_df['Label'] = "scabies"

################# Seborrheic ##########
seborrheic_test_df = pd.DataFrame()
seborrheic_test_df['Image'] = [seborrheic_test_path + '/' + img for img in seborrheic_test_files]
seborrheic_test_df['Label'] = "seborrheic"

################# Systemic ##########
systemic_test_df = pd.DataFrame()
systemic_test_df['Image'] = [systemic_test_path + '/' + img for img in systemic_test_files]
systemic_test_df['Label'] = "systemic"

################# Tinea ##########
tinea_test_df = pd.DataFrame()
tinea_test_df['Image'] = [tinea_test_path + '/' + img for img in tinea_test_files]
tinea_test_df['Label'] = "tinea"

################# Urticaria ##########
urticaria_test_df = pd.DataFrame()
urticaria_test_df['Image'] = [urticaria_test_path + '/' + img for img in urticaria_test_files]
urticaria_test_df['Label'] = "urticaria"

################# Vascular Tumors ##########
vascular_tumors_test_df = pd.DataFrame()
vascular_tumors_test_df['Image'] = [vascular_tumors_test_path + '/' + img for img in vascular_tumors_test_files]
vascular_tumors_test_df['Label'] = "vascular_tumors"

################# Vasculitis ##########
vasculitis_test_df = pd.DataFrame()
vasculitis_test_df['Image'] = [vasculitis_test_path + '/' + img for img in vasculitis_test_files]
vasculitis_test_df['Label'] = "vasculitis"

###########################################
###########################################

# Append all DataFrames to final_test_df
final_test_df = pd.concat([
    acne_test_df,
    actinic_test_df,
    atopic_test_df,
    bullous_test_df,
    cellulitis_test_df,
    eczema_test_df,
    exanthems_test_df,
    herpes_test_df,
    light_diseases_test_df,
    lupus_test_df,
    melanoma_test_df,
    poison_ivy_test_df,
    scabies_test_df,
    seborrheic_test_df,
    systemic_test_df,
    tinea_test_df,
    urticaria_test_df,
    vascular_tumors_test_df,
    vasculitis_test_df
], ignore_index=True)

print(f"Final Test DataFrame shape: {final_test_df.shape}")
train_data_gen  = ImageDataGenerator(
                                    rescale=1 / 255.0,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    validation_split=0.2,
                                    fill_mode='nearest')
test_data_gen = ImageDataGenerator(rescale=1 / 255.0)
batch_size = 8
train_generator = train_data_gen.flow_from_dataframe(
    dataframe=final_df,
    x_col="Image",
    y_col="Label",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",#sparse
    subset='training',
    shuffle=True,
    seed=42
)
valid_generator = train_data_gen.flow_from_dataframe(
    dataframe=final_df,
    x_col="Image",
    y_col="Label",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical", #sparse
    subset='validation',
    shuffle=True,
    seed=42
)
test_generator = test_data_gen.flow_from_dataframe(
    dataframe=final_test_df,
    x_col="Image",
    y_col="Label",
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
)
# VGG16 with Input shape of our Images
# Include Top is set to false to allow us to add more layers
from tensorflow.keras.applications import Xception
res = Xception(weights ='imagenet', include_top = False,
               input_shape = (224, 224, 3))

# Setting the trainable to false
res.trainable = False


x= res.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
# x = Dropout(0.5)(x)
x = Dense(512, activation ='relu')(x)
x = BatchNormalization()(x)
# x = Dropout(0.5)(x)

x = Dense(256, activation ='relu')(x)
x = BatchNormalization()(x)

x = Dense(19, activation ='softmax')(x) # Changed from 3 to 19
model = Model(res.input, x)
model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.001),  #'Adam'
              loss ="categorical_crossentropy",  #sparse_categorical_crossentropy
              metrics =["categorical_accuracy"])  #sparse_categorical_accuracy

model.summary()
custom_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    mode='min'
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define callbacks first


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
# Ensure models_dir and timestamp exist before creating checkpoint callback
if 'models_dir' not in globals():
    if IN_COLAB:
        models_dir = os.path.join(PROJECT_DIR, 'models')
    else:
        models_dir = os.path.join(os.path.dirname(__file__), 'models') if '__file__' in globals() else os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

model_checkpoint = ModelCheckpoint(
    os.path.join(models_dir, f'best_model_19_efficient_diseases_{timestamp}.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Compile the model first (if not already compiled)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully!")

# Now train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=valid_generator,
    callbacks=[  model_checkpoint , reduce_lr],
    verbose=1
)
# Create a models directory inside the project and save the trained models and metadata there
if IN_COLAB:
    models_dir = os.path.join(PROJECT_DIR, 'models')
else:
    models_dir = os.path.join(os.path.dirname(__file__), 'models') if '__file__' in globals() else os.path.join(os.getcwd(), 'models')
os.makedirs(models_dir, exist_ok=True)

# Save a timestamp with the model to track different training runs
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# The ModelCheckpoint callback already writes the best model to the current working directory.
# If it used a relative filename, move/rename it into the models folder (if present), otherwise
# ensure subsequent saves write to the models directory.
try:
    # Save final model in HDF5 format
    final_model_path = os.path.join(models_dir, 'final_model_19_efficient_diseases.h5')
    model.save(final_model_path)

    # Also save in TensorFlow SavedModel format
    saved_model_dir = os.path.join(models_dir, 'saved_model_19_efficient_diseases')
    model.save(saved_model_dir, save_format='tf')

    # Save class indices mapping so callers can map predictions back to labels
    import json
    class_indices_path = os.path.join(models_dir, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f, indent=4)

    print(f"Saved final HDF5 model to: {final_model_path}")
    print(f"Saved SavedModel to: {saved_model_dir}")
    print(f"Saved class indices to: {class_indices_path}")
except Exception as e:
    print(f"Warning: could not save model artifacts to {models_dir}: {e}")
#plot accuracy vs epoch
plt.plot(history.history['accuracy']) #sparse_categorical_accuracy
plt.plot(history.history['val_accuracy']) #val_sparse_categorical_accuracy
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# =============================================================================
# TESTING AND PREDICTION SECTION (AFTER MODEL LOSS PLOT)
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

# Get class names from the generator
class_names = list(test_generator.class_indices.keys())
print(f"Number of classes: {len(class_names)}")
print(f"Class names: {class_names}")

# Calculate accuracy
from sklearn.metrics import accuracy_score
xp_acc = accuracy_score(test_true, test_pred)
print(f"Xception Model Accuracy: {xp_acc * 100:.2f}%")

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
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0  # Normalize like training data
        
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

def test_random_images(num_images=5):
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
print(f"   Overall accuracy: {xp_acc * 100:.2f}%")

# =============================================================================
# RUN PREDICTIONS
# =============================================================================

# Test random images
test_random_images(num_images=5)

# Test specific images from each class
predict_specific_test_images()

# =============================================================================
# TEST YOUR OWN IMAGE
# =============================================================================

print(f"\n🖼️  TESTING YOUR OWN IMAGE")
print("=" * 40)

# Test the specific image you mentioned
test_image_path = r'E:\Desktop\biometric\biometric project dataset\train\Psoriasis pictures Lichen Planus and related diseases\08PsoriasisOnycholysis1.jpg'

if os.path.exists(test_image_path):
    print(f"Testing image: {os.path.basename(test_image_path)}")
    predicted_class, confidence, top_3_classes, top_3_confidences = predict_image_class(test_image_path, 'unknown')
    
    if predicted_class:
        print(f"\n🎯 FINAL PREDICTION:")
        print(f"   Image: {os.path.basename(test_image_path)}")
        print(f"   Predicted Disease: {predicted_class}")
        print(f"   Confidence: {confidence:.2f}%")
else:
    print(f"❌ Test image not found: {test_image_path}")
    print("Trying to find any test image from the dataset...")
    
    # Try to use any available test image
    if len(final_test_df) > 0:
        sample_image = final_test_df.iloc[0]
        test_image_path = sample_image['Image']
        true_label = sample_image['Label']
        print(f"Using sample image: {os.path.basename(test_image_path)}")
        predict_image_class(test_image_path, true_label)

print("\n✅ Testing and prediction completed successfully!")