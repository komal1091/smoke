import numpy as np
from ultralytics import YOLO
from PIL import Image
import os


def normalize_box(box, image_width, image_height):
    xmin, ymin, xmax, ymax = box
    return xmin * image_width, ymin * image_height, xmax * image_width, ymax * image_height

def calculate_iou(box_a, box_b):
    if np.isscalar(box_a):  # Check if box1 is a scalar (empty box)
        box_a = [0, 0, 0, 0]  # Initialize an empty box
    
    # Convert normalized box coordinates to pixel coordinates
    box_a = normalize_box(box_a, 240, 240)
    # box_b = normalize_box(box_b, image_width, image_height)

    # Extract coordinates for box A
    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    width_a = xmax_a - xmin_a
    height_a = ymax_a - ymin_a

    # Extract coordinates for box B
    xmin_b, ymin_b, xmax_b, ymax_b = box_b
    width_b = xmax_b - xmin_b
    height_b = ymax_b - ymin_b

    # Calculate intersection area
    x_min_intersect = max(xmin_a, xmin_b)
    y_min_intersect = max(ymin_a, ymin_b)
    x_max_intersect = min(xmax_a, xmax_b)
    y_max_intersect = min(ymax_a, ymax_b)

    intersection_area = max(0, x_max_intersect - x_min_intersect) * max(0, y_max_intersect - y_min_intersect)

    # Calculate union area
    area_a = width_a * height_a
    area_b = width_b * height_b
    union_area = area_a + area_b - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    print(box_a,'box1')
    print(box_b,'box2')
    print(iou,'iou')
    return iou


def evaluate_predictions(model, image_dir, annotation_dir=None):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    all_ious = []
    correct_predictions = 0  # Track images with at least one True Positive
    incorrect_prediction = 0
    total_images = len(os.listdir(image_dir))  # Count total images

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        img = Image.open(image_path)

        # Perform inference
        results = model(img)

        # Assuming ground truth annotations are available
        if annotation_dir:
            annotation_path = os.path.join(annotation_dir, os.path.splitext(image_name)[0] + '.txt')
            ground_truth = np.loadtxt(annotation_path)
            # print(ground_truth,'gt')

            # Extract predicted bounding boxes
            predicted_boxes = np.array([[box.xyxy[0] for box in result.boxes] for result in results])
            predicted_boxes = predicted_boxes[0][0]
            # print(predicted_boxes,'pdddddddddddd')

            # Compare predicted boxes with ground truth
            if len(predicted_boxes) == 0:  # Check if predicted_boxes is empty
                false_negatives += len(ground_truth)
                continue

            ious = calculate_iou(ground_truth[1:], predicted_boxes)
            # print(ious)
       
            if ious >= 0.5:
                true_positives += 1
                correct_predictions += 1  # Increment if there's at least one True Positive
            else:
                false_negatives += 1
                incorrect_prediction += 1


        false_positives += len(predicted_boxes) - true_positives

    # Compute accuracy metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = correct_predictions / total_images  # Calculate basic accuracy

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("avg_iou", sum(all_ious) / len(all_ious))
    print("Basic Accuracy:", accuracy)  

# Path to the directory containing test images
image_dir = '/home/codezeros/Documents/fire&smoke detection/Test/datasets/fire-8/test/images'
# Path to the directory containing ground truth annotations (if available)
annotation_dir = '/home/codezeros/Documents/fire&smoke detection/Test/datasets/fire-8/test/labels'
# Evaluate predictions
model = YOLO('/home/codezeros/Documents/fire&smoke detection/Test/best.pt')

evaluate_predictions(model, image_dir, annotation_dir)
