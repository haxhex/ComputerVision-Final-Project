import os
import cv2

# Define the path to the image directory
image_dir = "D:/ComputerVision-Final-Project/piece-detection-dataset/valid/images"
output_dir = "D:/ComputerVision-Final-Project/corners-dataset-genrator-output"

# Define the number of corners to label
num_corners = 4

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over the images in the image directory
for image_name in os.listdir(image_dir):
    # Load the image
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    
    # Create a window to display the image
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)

    # Create a list to store the corner coordinates
    corner_coordinates = []

    # Mouse callback function to capture the corner coordinates
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corner_coordinates.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", image)

    # Set the mouse callback function
    cv2.setMouseCallback("Image", mouse_callback)

    # Wait for the user to label the corners
    while len(corner_coordinates) < num_corners:
        cv2.waitKey(1)

    # Save the corner coordinates to a text file
    label_path = os.path.join(output_dir, image_name.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        for corner in corner_coordinates:
            f.write(f"{corner[0]} {corner[1]}\n")

    # Save the resized image with size 416x416
    resized_image_path = os.path.join(output_dir, image_name)
    cv2.imwrite(resized_image_path, image)

    # Close the image window
    cv2.destroyAllWindows()
