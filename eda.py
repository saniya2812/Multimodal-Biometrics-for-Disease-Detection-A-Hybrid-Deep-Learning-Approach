import cv2 
import os
import random
import matplotlib.pyplot as plt
import numpy as np


# FIX: Use a raw string (r'...') to avoid the SyntaxWarning for backslashes
data_path= r'E:\Desktop\biometric\archive (4)\train'

classes=os.listdir(data_path)
# FIX: Removed the incorrect indentation here.
dic={}
for i in classes:
    dic[i]= len(os.listdir(os.path.join(data_path,i)))
for key,value in dic.items():
    print(key,":",value,"\n")

plt.figure(figsize=(12, 8))
plt.bar(dic.keys(), dic.values())
plt.xticks(rotation=90)
plt.xlabel('Class')
plt.ylabel('Number of Files')
plt.title('Number of Files in Each Class')
plt.show()


plt.figure(figsize=(12, 8))
plt.pie(dic.values(), 
        labels=dic.keys(), 
        autopct='%1.1f%%', 
        startangle=140, 
        textprops={'fontsize': 10}) 

plt.axis('equal') 
plt.title('Distribution of Files in Each Class')
plt.show()

#CLASSES WITH MAXIMUM ELEMENTS

sorted_dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
top_classes = dict(list(sorted_dic.items())[:10])
for key, value in top_classes.items():
    print(key, ":", value, "\n")


train_data = []
val_data = []

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    file = os.listdir(folder_path)
    num_train = int(0.8 * len(file))
    files_train = random.sample(file, num_train)
    files_val = list(set(file) - set(files_train))
    
    for file in files_train:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224,224))
        train_data.append((img, folder))
        
    for file in files_val:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224,224))
        val_data.append((img, folder))

fig, axes = plt.subplots(3, 4, figsize=(12, 6))
plt.suptitle('LABELS OF EACH IMAGE')

for (img, label), ax in zip(random.sample(train_data, 12), axes.flatten()):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid(False)
    ax.set_title(label)
    ax.set_title(label, fontsize=6)
    # Ensure image is not None before converting color space
    if img is not None:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        # Handle case where image could not be read
        ax.text(0.5, 0.5, "Image Error", ha='center', va='center')


plt.show()


directories = {}
classes = []
# FIX: Use a raw string here too
for dirname , _ , files in os.walk(r'E:\Desktop\biometric\archive (4)\train'):
    # The rsplit line assumes a Unix-like path separator '/', which might be incorrect on Windows.
    # It's better to use os.path.basename or os.path.split.
    # For now, I'll use os.path.split to handle cross-platform directory name extraction.
    className = os.path.basename(dirname) 
    if (className != 'test' and className != 'train'): directories[className] = dirname
    classes.append(className)

# The logic to remove the first element assumes 'train' is always the first, 
# which depends on os.walk's order. It's safer to filter.
if 'train' in classes:
    classes.remove('train')
    
no_of_classes = len(classes)
print(classes)
print(no_of_classes)


def get_dimensions(filePath):
    img = cv2.imread(filePath)
    # Check if the image was read correctly
    if img is None:
        return (0, 0, 3) # Return a default shape if image is not found
    x , y , _ = img.shape
    return (x,y)

print(directories)



cols = 4
rows = (no_of_classes + cols - 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 5))

axs = axs.flatten()
heights_all = []
widths_all = []
for idx, (className, direc) in enumerate(directories.items()):
    x = []
    y = []
    for root, dirs, files in os.walk(direc):
        for img in files:
            h, w = get_dimensions(os.path.join(root, img))
            # Only process if dimensions are valid (not the error default of 0)
            if h > 0 and w > 0:
                widths_all.append(w)
                x.append(w)
                heights_all.append(h)
                y.append(h)
    
    # Ensure there is data to plot before calling scatter
    if x and y:
        axs[idx].scatter(x, y, color='red')
        axs[idx].set_title(className, fontsize=13)
        axs[idx].set_xlabel('Width', fontsize=12)
        axs[idx].set_ylabel('Height', fontsize=12)
        axs[idx].set_aspect('equal')
    else:
        axs[idx].set_title(f"{className} (No Data)", fontsize=13)


for i in range(idx + 1, len(axs)):
    fig.delaxes(axs[i])

fig.text(0.5, 0.04, 'Width', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Height', va='center', rotation='vertical', fontsize=14)

plt.tight_layout()
plt.show()


plt.scatter(widths_all, heights_all, color='red', s=100, alpha=0.5, edgecolors='w')

plt.xlabel('width')
plt.ylabel('heights')
plt.title('for all images')

plt.grid(True)

plt.show()