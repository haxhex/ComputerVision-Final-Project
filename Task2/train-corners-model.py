import os
import numpy as np
import tensorflow as tf

import cv2

# Define hyperparameters
batch_size = 32
epochs = 10

# Define the model architecture
# Creates a new sequential model object
model = tf.keras.models.Sequential([ 
    # Add a 2D convolutional layer with 32 filters of size 3x3 and ReLU activation function
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # Add a max pooling layer with a pool size of 2x2
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Add a 2D convolutional layer with 64 filters of size 3x3 and ReLU activation function
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Add a max pooling layer with a pool size of 2x2
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Add a flatten layer to the model, which flattens the output of the previous layer into a 1D vector
    tf.keras.layers.Flatten(),
    # Add a fully connected layer with 64 units and ReLU activation function
    tf.keras.layers.Dense(64, activation='relu'),
    # Add another fully connected layer with 8 units to the model, 
    # which corresponds to the 4 corners of the chessboard (each corner has 2 coordinates).
    tf.keras.layers.Dense(8)  # 8 outputs for the coordinates of 4 corners
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

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

# Load the training, testing, and validation datasets
train_images, train_labels = load_dataset('D:/ComputerVision-Final-Project/corner-detection-dataset/train')
test_images, test_labels = load_dataset('D:/ComputerVision-Final-Project/corner-detection-dataset/test')
valid_images, valid_labels = load_dataset('D:/ComputerVision-Final-Project/corner-detection-dataset/valid')

# Normalize the image pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0
valid_images = valid_images / 255.0

# Train the model
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(valid_images, valid_labels))

# Save the model
model.save('D:/ComputerVision-Final-Project/corner-detection-models/chessboard_corners_model2023.h5')

# Make predictions
predictions = model.predict(test_images)

# Print sample predictions
for i in range(5):
    print('Image:', test_images[i])
    print('True coordinates:', test_labels[i])
    print('Predicted coordinates:', predictions[i])
    print()

# Visualize predicted corners and real corners on test images
for i in range(len(test_images)):
    image = test_images[i]
    label_coordinates = test_labels[i]
    predicted_coordinates = predictions[i]

    # Reshape the coordinates to (4, 2) shape - assuming 4 corners
    label_coordinates = label_coordinates.reshape((4, 2))
    predicted_coordinates = predicted_coordinates.reshape((4, 2))

    # Convert the image to uint8 data type
    image = np.uint8(image * 255)

    # Draw circles for the label coordinates (real corners) in blue
    for corner in label_coordinates:
        corner = tuple(map(int, corner))
        cv2.circle(image, corner, 5, (255, 0, 0), -1)

    # Draw circles for the predicted coordinates in green
    for corner in predicted_coordinates:
        corner = tuple(map(int, corner))
        cv2.circle(image, corner, 5, (0, 255, 0), -1)

    # Show the image with corners
    cv2.imshow('Test Image with Corners', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()