import os
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('D:/ComputerVision-Final-Project/corner-detection-models/chessboard_corners_model1002.h5')
# Load the dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    
    # Read each image and its corresponding coordinates
    for image_filename in os.listdir(dataset_path):
        if image_filename.endswith('.jpg'):
            image_path = os.path.join(dataset_path, image_filename)
            text_filename = image_filename.replace('.jpg', '.txt')
            text_path = os.path.join(dataset_path, text_filename)
            
            # Load the image
            image = tf.keras.preprocessing.image.load_img(image_path)
            image = tf.keras.preprocessing.image.img_to_array(image)
            
            # Load the coordinates from the text file
            with open(text_path, 'r') as f:
                coordinates = f.readlines()
                coordinates = [list(map(float, coord.strip().split())) for coord in coordinates]
                coordinates = [coord for sublist in coordinates for coord in sublist]
            
            # Add the image and coordinates to the dataset
            images.append(image)
            labels.append(coordinates)
    
    # Convert the dataset to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Load the test and training datasets
test_images, test_labels = load_dataset('D:/ComputerVision-Final-Project/corner-detection-dataset/test')
train_images, train_labels = load_dataset('D:/ComputerVision-Final-Project/corner-detection-dataset/train')

# Normalize the image pixel values to [0, 1]
test_images = test_images / 255.0
train_images = train_images / 255.0

# Evaluate the model on the test data
test_loss = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)

# Make predictions on the test data
predictions = model.predict(test_images)

# Calculate the accuracy
test_accuracy = np.mean(np.abs(predictions - test_labels) < 5)
print('Test Accuracy:', test_accuracy)

# Evaluate the model on the training data
train_loss = model.evaluate(train_images, train_labels)
print('Training Loss:', train_loss)

# Make predictions on the training data
train_predictions = model.predict(train_images)

# Calculate the accuracy
train_accuracy = np.mean(np.abs(train_predictions - train_labels) < 5)
print('Training Accuracy:', train_accuracy)
