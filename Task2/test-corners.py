import os
import numpy as np
import tensorflow as tf
import cv2

# Load the saved model
# chessboard_corners_model.h5
# chessboard_corners_model2.h5
# chessboard_corners_model6.h5
# chessboard_corners_model1001.h5
# chessboard_corners_model1002.h5
# chessboard_corners_model1003.h5
model = tf.keras.models.load_model('D:/ComputerVision-Final-Project/corner-detection-models/chessboard_corners_model1002.h5')

# Load and preprocess the new image
image_path = 'D:/ComputerVision-Final-Project/corner-detection-dataset/test/685b860d412b91f5d4f7f9e643b84452_jpg.rf.2d78193e4021ae5ffb49ecd1060bebd7.jpg'
image = tf.keras.preprocessing.image.load_img(image_path)
image = tf.keras.preprocessing.image.img_to_array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Make predictions on the new image
predictions = model.predict(image)

# Process the predicted coordinates
predicted_coordinates = predictions.reshape((4, 2))

# Visualize the predicted corners on the image
image = cv2.imread(image_path)
for corner in predicted_coordinates:
    corner = tuple(map(int, corner))
    cv2.circle(image, corner, 5, (0, 255, 0), -1)
cv2.imshow('New Image with Predicted Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
