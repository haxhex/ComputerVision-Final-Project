import torch
import cv2

def main():

    # Load YOLOv5 from Ultralytics with PyTorch Hub
    # best1001.pt
    # best1010.pt
    # best1011.pt
    model = torch.hub.load("ultralytics/yolov5", "custom", path="D:/ComputerVision-Final-Project/piece-detection-models/best1011.pt")
    model.eval()

    # Load the image
    image = cv2.imread("D:/ComputerVision-Final-Project/piece-detection-dataset/test/images/685b860d412b91f5d4f7f9e643b84452_jpg.rf.2d78193e4021ae5ffb49ecd1060bebd7.jpg")

    # Convert the image to the expected format (BGR to RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the inference
    results = model(image)

    # Display the image with bounding boxes
    image_with_boxes = results.render()[0]
    cv2.imshow("Image with Predicted Boxes", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
