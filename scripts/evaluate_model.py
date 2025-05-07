# evaluate_model.py
import torch

def evaluate_model(model, val_loader, device, iou_threshold=0.5):
    model.eval()
    total_boxes = 0
    correct_boxes = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = [label.to(device) for label in labels]

            outputs = model(images)

            for i in range(outputs.shape[0]):
                predicted_boxes = outputs[i]
                true_boxes = labels[i]

                correct_boxes += compare_boxes(predicted_boxes, true_boxes, iou_threshold)
                total_boxes += len(true_boxes)

    accuracy = 100 * correct_boxes / total_boxes if total_boxes > 0 else 0
    print(f"Model Evaluation - IoU Accuracy: {accuracy:.2f}%")

def compare_boxes(predicted_boxes, true_boxes, iou_threshold):
    correct = 0
    for true_box in true_boxes:
        best_iou = 0
        for predicted_box in predicted_boxes:
            iou = compute_iou(true_box, predicted_box)
            if iou > best_iou:
                best_iou = iou
        if best_iou >= iou_threshold:
            correct += 1
    return correct

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

