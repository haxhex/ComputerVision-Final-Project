# Import statements for the required libraries and modules
import pandas as pd
import torch
import cv2
import numpy as np
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import tensorflow as tf

# axis=0 refers to the vertical axis (i.e., the rows) 
# and axis=1 refers to the horizontal axis

def order_points(pts):
	# order a list of 4 coordinates:
	# 0: top-left,
	# 1: top-right
	# 2: bottom-right,
	# 3: bottom-left
	# Create an empty NumPy array with 4 rows and 2 columns to store the ordered points
	rect = np.zeros((4, 2), dtype = "float32")
    # Calculate the sum of x and y coordinates for each point
	s = pts.sum(axis = 1)
    # Find the point with the smallest sum of coordinates and store it as the top-left point
	rect[0] = pts[np.argmin(s)]
    # Find the point with the largest sum of coordinates and store it as the bottom-right point
	rect[2] = pts[np.argmax(s)]

    # Calculate the difference between the x and y coordinates for each point	
	diff = np.diff(pts, axis = 1)
    # Find the point with the smallest difference between its x and y coordinates and store it as the top-right point
	rect[1] = pts[np.argmin(diff)]
    # Find the point with the largest difference between its x and y coordinates and store it as the bottom-left point 
	rect[3] = pts[np.argmax(diff)]
 
    # Return the ordered points [top-left, top-right, bottom-right, bottom-left]
	return rect

# Calculates chessboard grid
def plot_grid_on_transformed_image(image):
    # Define the corners of the image as a 4x2 NumPy array
	corners = np.array([[0,0], 
					[image.shape[0], 0], 
					[0, image.shape[1]], 
					[image.shape[0], image.shape[1]]])
    # Order the corners of the image in [top-left, top-right, bottom-right, bottom-left]
	corners = order_points(corners)
    # Create a new figure with a specific size and resolution
	figure(figsize=(10, 10), dpi=80)
	
	# put image on grid
	implot = plt.imshow(image)
 
    # Extract the ordered corners of the image
	TL = corners[0]
	BL = corners[3]
	TR = corners[1]
	BR = corners[2]

    # Define a function to interpolate points between two given points
	def interpolate(xy0, xy1):
   		# Unpack the input points into their x and y coordinates
		x0,y0 = xy0
		x1,y1 = xy1
	    # Calculate the step size for each interpolated point
		dx = (x1-x0) / 8
		dy = (y1-y0) / 8
	    # Create a list of 9 interpolated points
		pts = [(x0+i*dx,y0+i*dy) for i in range(9)]
	    # Return the list of interpolated points
		return pts

    # Interpolate points between the top-left and top-right corners
	ptsT = interpolate( TL, TR )
    # Interpolate points between the top-left and bottom-left corners
	ptsL = interpolate( TL, BL )
    # Interpolate points between the top-left and bottom-left corners
	ptsR = interpolate( TR, BR )
    # Interpolate points between the top-left and bottom-left corners
	ptsB = interpolate( BL, BR )
 
    # Plot the horizontal lines of the grid
	for a,b in zip(ptsL, ptsR):
		plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )

    # Plot the vertical lines of the grid
	for a,b in zip(ptsT, ptsB):
		plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
	
     # Turn off the axis labels and ticks	
	plt.axis('off')

    # Save the plot as an image file
	plt.savefig('D:/ComputerVision-Final-Project/Task3/chessboard_transformed_with_grid.jpg')
 
    # Return the interpolated points for each side of the grid
	return ptsT, ptsL, ptsR, ptsB

def calculate_iou(box_1, box_2):
    # Create polygons from the input bounding boxes
	poly_1 = Polygon(box_1)
	poly_2 = Polygon(box_2)
  	# Calculate the intersection area between the polygons
    # Calculate the union area between the 
    # Calculate the IoU as the intersection area divided by the union area
	iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    # Return the IoU
	return iou

def connect_square_to_detection(detections, square):
    # Define a dictionary that maps detection indices to chess piece names
	di = {0: 'b', 1: 'k', 2: 'n',
	  3: 'p', 4: 'q', 5: 'r', 
	  6: 'B', 7: 'K', 8: 'N',
	  9: 'P', 10: 'Q', 11: 'R'}
    # Initialize a list to store the IoU values
	list_of_iou=[]
    # Loop through each detection box
	for box in detections:
     
        # Extract the x and y coordinates of the box corners
		box_x1 = box[0]
		box_y1 = box[1]
		# --------------
		box_x2 = box[2]
		box_y2 = box[1]
		# --------------
		box_x3 = box[2]
		box_y3 = box[3]
		# --------------
		box_x4 = box[0]
		box_y4 = box[3]
		
		# Cut high pieces    
		# If the height of a box is greater than 45 pixels, 
  		# the function adds 25 pixels to the y coordinates of 
    	# the top corners of the box to cut off the top of the chess piece.
		if box_y4 - box_y1 > 45:
			box_complete = np.array([[box_x1,box_y1+25], [box_x2, box_y2+25], [box_x3, box_y3], [box_x4, box_y4]])
		else:
			box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])
			
		# until here
        # Calculate the IoU between the detection box and the square
		list_of_iou.append(calculate_iou(box_complete, square))

    # Find the index of the detection box with the highest IoU
	num = list_of_iou.index(max(list_of_iou))

    # Map the detection index to a chess piece name using the 'di' dictionary
	piece = detections[num][5] -1
 
	# If the highest IoU is above a threshold, return the corresponding chess piece name
	if max(list_of_iou) > 0.2:
		return di[piece]

	# Otherwise, return "empty"
	else:
		piece = "empty"
		return piece

def print_link(ptsT, ptsL, detections):
	# Calculate the grid
    # Define the x and y coordinates of the chessboard squares
	xA = ptsT[0][0]
	xB = ptsT[1][0]
	xC = ptsT[2][0]
	xD = ptsT[3][0]
	xE = ptsT[4][0]
	xF = ptsT[5][0]
	xG = ptsT[6][0]
	xH = ptsT[7][0]
	xI = ptsT[8][0]
    # --------------
	y9 = ptsL[0][1]
	y8 = ptsL[1][1] 
	y7 = ptsL[2][1] 
	y6 = ptsL[3][1]  
	y5 = ptsL[4][1]  
	y4 = ptsL[5][1] 
	y3 = ptsL[6][1]  
	y2 = ptsL[7][1] 
	y1 = ptsL[8][1] 

	# Calculate all the squares
    # Define the coordinates of each chessboard square
	a8 = np.array([[xA,y9], [xB, y9], [xB, y8], [xA, y8]])
	a7 = np.array([[xA,y8], [xB, y8], [xB, y7], [xA, y7]])
	a6 = np.array([[xA,y7], [xB, y7], [xB, y6], [xA, y6]])
	a5 = np.array([[xA,y6], [xB, y6], [xB, y5], [xA, y5]])
	a4 = np.array([[xA,y5], [xB, y5], [xB, y4], [xA, y4]])
	a3 = np.array([[xA,y4], [xB, y4], [xB, y3], [xA, y3]])
	a2 = np.array([[xA,y3], [xB, y3], [xB, y2], [xA, y2]])
	a1 = np.array([[xA,y2], [xB, y2], [xB, y1], [xA, y1]])

	b8 = np.array([[xB,y9], [xC, y9], [xC, y8], [xB, y8]])
	b7 = np.array([[xB,y8], [xC, y8], [xC, y7], [xB, y7]])
	b6 = np.array([[xB,y7], [xC, y7], [xC, y6], [xB, y6]])
	b5 = np.array([[xB,y6], [xC, y6], [xC, y5], [xB, y5]])
	b4 = np.array([[xB,y5], [xC, y5], [xC, y4], [xB, y4]])
	b3 = np.array([[xB,y4], [xC, y4], [xC, y3], [xB, y3]])
	b2 = np.array([[xB,y3], [xC, y3], [xC, y2], [xB, y2]])
	b1 = np.array([[xB,y2], [xC, y2], [xC, y1], [xB, y1]])

	c8 = np.array([[xC,y9], [xD, y9], [xD, y8], [xC, y8]])
	c7 = np.array([[xC,y8], [xD, y8], [xD, y7], [xC, y7]])
	c6 = np.array([[xC,y7], [xD, y7], [xD, y6], [xC, y6]])
	c5 = np.array([[xC,y6], [xD, y6], [xD, y5], [xC, y5]])
	c4 = np.array([[xC,y5], [xD, y5], [xD, y4], [xC, y4]])
	c3 = np.array([[xC,y4], [xD, y4], [xD, y3], [xC, y3]])
	c2 = np.array([[xC,y3], [xD, y3], [xD, y2], [xC, y2]])
	c1 = np.array([[xC,y2], [xD, y2], [xD, y1], [xC, y1]])

	d8 = np.array([[xD,y9], [xE, y9], [xE, y8], [xD, y8]])
	d7 = np.array([[xD,y8], [xE, y8], [xE, y7], [xD, y7]])
	d6 = np.array([[xD,y7], [xE, y7], [xE, y6], [xD, y6]])
	d5 = np.array([[xD,y6], [xE, y6], [xE, y5], [xD, y5]])
	d4 = np.array([[xD,y5], [xE, y5], [xE, y4], [xD, y4]])
	d3 = np.array([[xD,y4], [xE, y4], [xE, y3], [xD, y3]])
	d2 = np.array([[xD,y3], [xE, y3], [xE, y2], [xD, y2]])
	d1 = np.array([[xD,y2], [xE, y2], [xE, y1], [xD, y1]])

	e8 = np.array([[xE,y9], [xF, y9], [xF, y8], [xE, y8]])
	e7 = np.array([[xE,y8], [xF, y8], [xF, y7], [xE, y7]])
	e6 = np.array([[xE,y7], [xF, y7], [xF, y6], [xE, y6]])
	e5 = np.array([[xE,y6], [xF, y6], [xF, y5], [xE, y5]])
	e4 = np.array([[xE,y5], [xF, y5], [xF, y4], [xE, y4]])
	e3 = np.array([[xE,y4], [xF, y4], [xF, y3], [xE, y3]])
	e2 = np.array([[xE,y3], [xF, y3], [xF, y2], [xE, y2]])
	e1 = np.array([[xE,y2], [xF, y2], [xF, y1], [xE, y1]])

	f8 = np.array([[xF,y9], [xG, y9], [xG, y8], [xF, y8]])
	f7 = np.array([[xF,y8], [xG, y8], [xG, y7], [xF, y7]])
	f6 = np.array([[xF,y7], [xG, y7], [xG, y6], [xF, y6]])
	f5 = np.array([[xF,y6], [xG, y6], [xG, y5], [xF, y5]])
	f4 = np.array([[xF,y5], [xG, y5], [xG, y4], [xF, y4]])
	f3 = np.array([[xF,y4], [xG, y4], [xG, y3], [xF, y3]])
	f2 = np.array([[xF,y3], [xG, y3], [xG, y2], [xF, y2]])
	f1 = np.array([[xF,y2], [xG, y2], [xG, y1], [xF, y1]])

	g8 = np.array([[xG,y9], [xH, y9], [xH, y8], [xG, y8]])
	g7 = np.array([[xG,y8], [xH, y8], [xH, y7], [xG, y7]])
	g6 = np.array([[xG,y7], [xH, y7], [xH, y6], [xG, y6]])
	g5 = np.array([[xG,y6], [xH, y6], [xH, y5], [xG, y5]])
	g4 = np.array([[xG,y5], [xH, y5], [xH, y4], [xG, y4]])
	g3 = np.array([[xG,y4], [xH, y4], [xH, y3], [xG, y3]])
	g2 = np.array([[xG,y3], [xH, y3], [xH, y2], [xG, y2]])
	g1 = np.array([[xG,y2], [xH, y2], [xH, y1], [xG, y1]])

	h8 = np.array([[xH,y9], [xI, y9], [xI, y8], [xH, y8]])
	h7 = np.array([[xH,y8], [xI, y8], [xI, y7], [xH, y7]])
	h6 = np.array([[xH,y7], [xI, y7], [xI, y6], [xH, y6]])
	h5 = np.array([[xH,y6], [xI, y6], [xI, y5], [xH, y5]])
	h4 = np.array([[xH,y5], [xI, y5], [xI, y4], [xH, y4]])
	h3 = np.array([[xH,y4], [xI, y4], [xI, y3], [xH, y3]])
	h2 = np.array([[xH,y3], [xI, y3], [xI, y2], [xH, y2]])
	h1 = np.array([[xH,y2], [xI, y2], [xI, y1], [xH, y1]])

	# Transforms the squares to write FEN
    # Create a 2D array of the chessboard squares
	FEN_annotation = [[a8, b8, c8, d8, e8, f8, g8, h8],
					[a7, b7, c7, d7, e7, f7, g7, h7],
					[a6, b6, c6, d6, e6, f6, g6, h6],
					[a5, b5, c5, d5, e5, f5, g5, h5],
					[a4, b4, c4, d4, e4, f4, g4, h4],
					[a3, b3, c3, d3, e3, f3, g3, h3],
					[a2, b2, c2, d2, e2, f2, g2, h2],
					[a1, b1, c1, d1, e1, f1, g1, h1]]


	board_FEN = []
	corrected_FEN = []
	complete_board_FEN = []

    # Loop through each row of the chessboard
	for line in FEN_annotation:
		line_to_FEN = []
        # Loop through each square in the row
		for square in line:
			# Determine which chess piece is on the square
			piece_on_square = connect_square_to_detection(detections, square)    
			line_to_FEN.append(piece_on_square)
        # Replace any empty squares with a '1'
		corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
  
		print(corrected_FEN)
		# Append the row to the board FEN
		board_FEN.append(corrected_FEN)
  
    # Rotate the board FEN 90 degrees clockwise
	board_FEN = np.rot90(board_FEN)
    # Convert the board FEN to a list of strings
	complete_board_FEN = [''.join(line) for line in board_FEN] 
 
    # Join the strings with '/' to create the FEN string
	to_FEN = '/'.join(complete_board_FEN)
 
    # Print the link to the Lichess website with the FEN string
	print("https://lichess.org/analysis/"+to_FEN)

def non_max_suppression_slow(boxes, overlapThresh):
	# If there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# Initialize the list of picked indexes
	pick = []
	# Grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	z  = boxes[:,4]
	# Compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# Keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# Grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# Loop over all indexes in the indexes list
		for pos in range(0, last):
			# Grab the current index
			j = idxs[pos]
			# Find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# Compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# Compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# If there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# Delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# Return only the bounding boxes that were picked
	return boxes[pick]
	

def main():
	# Load the trained model
	model = tf.keras.models.load_model('D:/ComputerVision-Final-Project/corner-detection-models/chessboard_corners_model1010.h5')
	
	# Load the chessboard image
	chessboard_image = cv2.imread('D:/ComputerVision-Final-Project/corner-detection-dataset/test/a3863d0be6002c21b20ac88817b2c56f_jpg.rf.0413d5178136ace55f588df9556c060a.jpg')

	# Preprocess the image
    # Normalize the pixel values of the image to be between 0 and 1
    # that help with model convergence and accuracy
	preprocessed_image = chessboard_image / 255.0
 
    # Adds an extra dimension to the image array to make it compatible with 
    # the input shape expected by the model
    # Shape of preprocessed_image before expand: (height, width, channels)
    # Shape of preprocessed_image after expand: (1, height, width, channels) = (batch_size, height, width, channels)
    # The 1 represents the batch size, indicating that there is a single image in the batch.
    # The batch dimension represents the number of samples in the batch. 
    # The axis=0 argument specifies that the extra dimension should be 
    # added as the first dimension of the array, which corresponds to the batch size. 
	preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

	# Make predictions
	predictions = model.predict(preprocessed_image)

    # The shape of predictions is (1, 8). 
    # The first dimension corresponds to the batch size, 
    # which is 1 in this case, and the second dimension corresponds to 
    # the 8 predicted corner coordinates (4 corners, each with x and y coordinates).
	# Reshape the predicted corners
	predicted_corners = predictions.reshape((4, 2))

	# Define the target corners for perspective transformation
	target_corners = np.array([[0, 0], [300, 0], [0, 300], [300, 300]], dtype=np.float32)

	# Get the perspective transformation matrix
	perspective_matrix = cv2.getPerspectiveTransform(predicted_corners, target_corners)

	# Apply the perspective transformation to the chessboard image
	transformed_image = cv2.warpPerspective(chessboard_image, perspective_matrix, (300, 300))
		
	# Load YOLOv5 from Ultralytics with PyTorch Hub
	model = torch.hub.load("ultralytics/yolov5", "custom", path="D:/ComputerVision-Final-Project/piece-detection-models/best1010.pt")
	model.eval()

	# Convert the image to the expected format (BGR to RGB)
	image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

	# Perform the inference
	results = model(image)

    # Get the grid points on the transformed image
	ptsT, ptsL, ptsR, ptsB = plot_grid_on_transformed_image(image)

    # Perform non-maximum suppression on the detection results
    # Extracts the bounding boxes and their corresponding confidence scores
	detections = results.xyxy
	t_detections = detections[0]
 
    # Corresponds to the detected bounding boxes and confidence scores for the
    # first image in the batch
	t_detections = t_detections.detach().numpy()

    # Apply to the bounding boxes with a threshold of 0.5 to remove overlapping bounding boxes
	suppedB = non_max_suppression_slow(t_detections,0.5)
 
   # Print the original and suppressed detection results
	print("Original detection results:\n", t_detections)
	print("Suppressed detection results:\n", suppedB)
 
	print_link (ptsT, ptsL, suppedB)
	# Display the image with bounding boxes

	# Process the detection results
	# Convert suppedB list to a DataFrame
	detections = pd.DataFrame(suppedB, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"])

	print(model.names)
	# Print the converted DataFrame
	print(detections)
	# Retrieve label names from the model
	label_names = model.module.names if hasattr(model, 'module') else model.names

	bounding_boxes = []
	label_names_list = []
	# Iterate over the detections and extract the label number, bounding box coordinates, and label name
	for index, detection in detections.iterrows():
		label_number = int(detection["class"])
		x1, y1, x2, y2 = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
		
		# Convert the coordinates to the desired format
		bounding_box = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
		
		# Add the bounding box and label name to the respective lists
		bounding_boxes.append(bounding_box)
		label_names_list.append(label_names[label_number])
		
	# Apply the perspective transformation to each bounding box individually
	transformed_bounding_boxes = []
	for bbox_points in bounding_boxes:

		# Apply the perspective transformation to the bounding box points		
		transformed_bbox_points = bbox_points
		transformed_bbox_points = transformed_bbox_points.astype(np.int32)
		transformed_bounding_boxes.append(transformed_bbox_points)

	# Draw the transformed bounding boxes on the transformed image along with label names
	for bbox_points, label_name in zip(transformed_bounding_boxes, label_names_list):
		cv2.polylines(transformed_image, [bbox_points], True, (0, 255, 0), thickness=2)
		cv2.putText(transformed_image, label_name, (bbox_points[0][0][0], bbox_points[0][0][1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# Display the transformed image with the bounding boxes and label names
	cv2.imshow('Transformed Image', transformed_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()