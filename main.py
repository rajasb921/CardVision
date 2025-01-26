''''
- Use this program to test the classification functionality of the model
- Choose either 10, 20, or all 52 cards to test
- The program will apply augmentation to 50% of the selected cards. This ensures that the model is robust to different orientations, 
  brightness levels, and shifts
- The program will display the image and the classification for all cards
- Will also display a confusion matrix at the end
'''

import cv2
import random
import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Helper function to load images and labels
def load_images_and_labels(img_dir, num_cards=None):
    images = []
    labels = []

    # Generate labelMap
    denominations = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['C', 'H', 'S', 'D']
    labelMap = {f"{denom}{suit}": idx for idx, (denom, suit) in enumerate((d, s) for d in denominations for s in suits)}

    # Get all valid image filenames
    valid_images = [
        filename for filename in os.listdir(img_dir) 
        if filename.endswith('.jpg') and filename[:2] in labelMap
    ]

    if num_cards is not None:
        if num_cards > len(valid_images):
            print(f"Warning: Requested {num_cards} cards, but only {len(valid_images)} available.")
            num_cards = len(valid_images)
        
        # Randomly select images
        selected_images = random.sample(valid_images, num_cards)
    else:
        selected_images = valid_images

    # Determine how many images to augment (50% of selected images)
    num_augment = num_cards // 2

    # Randomly select which images to augment
    augment_indices = random.sample(range(num_cards), num_augment)

    for i, filename in enumerate(selected_images):

        label_name = filename[:2]
        label = labelMap[label_name]

        # Read image
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
        img = img / 255.0

        # If this image is selected for augmentation
        if i in augment_indices:
            # Choose a random augmentation
            aug_type = random.choice(['brightness', 'shift'])
            
            if aug_type == 'brightness':
                factor = random.choice([0.9, 1.1])
                img = np.clip(img * factor, 0, 1)
            
            else:  # shift
                h_shift = random.choice([2, -2])
                v_shift = random.choice([2, -2])
                M = np.float32([[1, 0, h_shift], [0, 1, v_shift]])
                img = cv2.warpAffine(img, M, (128, 128), flags=cv2.INTER_LINEAR)

        # Append image
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Helper function to display images and predictions
def predict_labels(images, true_labels):
    model = load_model('./mobilenet_card_classifier_model.keras')  # Load the model
    predictions = model.predict(images)  # Get model predictions

    os.system('cls' if os.name == 'nt' else 'clear')

    # Get predicted labels
    pred_labels = np.argmax(predictions, axis=1)

    # Generate labelMap
    denominations = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['C', 'H', 'S', 'D']
    labelMap = {idx: f"{denom}{suit}" for idx, (denom, suit) in enumerate((d, s) for d in denominations for s in suits)}

    # Create a figure with subplots
    num_images = len(images)
    cols = min(5, num_images)  # Max 5 columns
    rows = (num_images + cols - 1) // cols  # Calculate necessary rows
    plt.figure(figsize=(15, 3 * rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        
        # Get predicted and true labels
        pred_label = labelMap[pred_labels[i]]
        true_label = labelMap[true_labels[i]]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return pred_labels

# Helper function to display confusion matrix
def display_confusion_matrix(true_labels, pred_labels):
    # Generate full labelMap
    denominations = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suits = ['C', 'H', 'S', 'D']
    full_labelMap = {idx: f"{denom}{suit}" for idx, (denom, suit) in enumerate((d, s) for d in denominations for s in suits)}


    # Create a full confusion matrix with zeros for all 52 possible classes
    cm = np.zeros((len(full_labelMap), len(full_labelMap)), dtype=int)

    # Fill in the confusion matrix for the actual predictions
    for true, pred in zip(true_labels, pred_labels):
        cm[true, pred] += 1

    
    plt.figure(figsize=(20, 16))

    # Create heatmap of the confusion matrix
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',     
                cmap='Blues',  
                xticklabels=[full_labelMap[i] for i in range(len(full_labelMap))],
                yticklabels=[full_labelMap[i] for i in range(len(full_labelMap))])

    # Set labels and title
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix for Card Classification', fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

    # Calculate and print overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f'Overall Accuracy: {accuracy:.2%}')

def main():
    img_dir = './cards/'  # Directory containing card images
    num_cards = 0
    os.system('cls' if os.name == 'nt' else 'clear')
    while num_cards not in [10, 20, 52]:
        num_cards = int(input("Enter number of cards to test (10, 20, or 52): "))  \

    images, true_labels = load_images_and_labels(img_dir, num_cards)
    pred_labels = predict_labels(images, true_labels)
    display_confusion_matrix(true_labels, pred_labels)

if __name__ == '__main__':
    main()