from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

model = YOLO('/home/codezeros/Documents/fire&smoke detection/Test/best.pt')
# Function to calculate IoU
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    # print(box1)
    # print("----------------------------------------------------------------")
    # print(box2)
    # print("----------------------------------------------------------------")
    # print(box1[0][1],box1[0][2],box1[0][2],box1[0][3])
    # print("----------------------------------------------------------------")
    # print(box2[0],box2[1],box2[2],box2[3])
    # x1 = max(box1[0][0], box2[0])
    # y1 = max(box1[0][1], box2[1])
    # x2 = min(box1[0][2], box2[2])
    # y2 = min(box1[0][3], box2[3])
    # # Calculate area of intersection rectangle
    # intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # # Calculate area of both bounding boxes
    # box1_area = (box1[0][2] - box1[0][0] + 1) * (box1[0][3] - box1[0][1] + 1)
    # box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # # Calculate IoU
    # iou = intersection_area / float(box1_area + box2_area - intersection_area)
    # return iou
    # print(box1[1],box1[2],box1[3],box1[4],'!!!!!!!!!!!!!!!!!!!!!!1')
    # print(box1)
    # print(box2)
    if np.isscalar(box1):
        box1 = np.array([0, 0, 0, 0,0])
    print(box1,'^^^^^^^^^^^^^^^^^66')
    x_left = max(box1[1], box2[0])
    y_top = max(box1[2], box2[1])
    x_right = min(box1[3], box2[2])
    y_bottom = min(box1[4], box2[3])
    # if x_right < x_left or y_bottom < y_top:
    #     return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
    box1_area = (box1[3] - box1[1]) * (box1[4] - box1[2])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    # Compute the IoU
    if union_area == 0:
        return 0.0
    iou = intersection_area / union_area
    print(iou)
    return iou
#Function to evaluate predictions
def evaluate_predictions(model, image_dir, annotation_dir=None):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_ious=[]
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        img = Image.open(image_path)
        # Perform inference
        results = model(img)
        # Assuming ground truth annotations are available
        if annotation_dir:
            annotation_path = os.path.join(annotation_dir, os.path.splitext(image_name)[0] + '.txt')
            ground_truth = np.loadtxt(annotation_path)
            # Extract predicted bounding boxes
            # for i in results:
            #     print(i.boxes.xyxy)
            predicted_boxes = np.array([[box.xyxy[0] for box in result.boxes] for result in results])
            # predicted_boxes = list(predicted_boxes[0][0])
            print(len(predicted_boxes),'lennnnnnnnnnnn')
            
            # print(predicted_boxes, type(predicted_boxes),'pd')

            # print("---------------------------------------")
            # print(predicted_boxes[0])
            # Compare predicted boxes with ground truth
            if len(predicted_boxes) == 0:  # Check if predicted_boxes is empty
                false_negatives += len(ground_truth)
                continue
            # Compare predicted boxes with ground truth
            for gt_box in ground_truth:
                ious = [calculate_iou(gt_box, pred_box) for pred_box in predicted_boxes[0]]
                if len(ious) == 0:  # Check if ious is empty
                    false_negatives += 1
                    continue
                max_iou = max(ious)
                all_ious.append(max_iou)
                if max_iou >= 0.5:
                    true_positives += 1
                else:
                    false_negatives += 1
                # print(max_iou,'*******************')
                # print(ious,'!!!!!!!!!!!!!!!!!!!!')
            false_positives += len(predicted_boxes) - true_positives
            # print(ious)
            # print(max_iou)
    # Compute accuracy metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("avg_iou",sum(ious)/len(ious))
    # print(all_ious)
# Path to the directory containing test images
image_dir = '/home/codezeros/Documents/fire&smoke detection/Test/datasets/fire-8/test/images'
# Path to the directory containing ground truth annotations (if available)
annotation_dir = '/home/codezeros/Documents/fire&smoke detection/Test/datasets/fire-8/test/labels'
# Evaluate predictions
evaluate_predictions(model, image_dir, annotation_dir)















