from torchvision import transforms
from PIL import Image
from numpy import asarray
from os.path import join
from train_classifier import get_model_instance_segmentation, parse_bounding_box, parse_labels
import pandas as pd
import torch
import cv2
import getopt
import sys

inverse_label_map = {
    1: "Wearing Mask Correctly",
    2: "Wearing Mask Incorrectly",
    3: "Not Wearing Mask"
}

color_map = {
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (0, 0, 255)
}

# Calculate percentage overlap between two bounding boxes
def calculate_box_overlap(box0, box1):
    xmin0, ymin0, xmax0, ymax0 = box0
    xmin1, ymin1, xmax1, ymax1 = box1
    # Calculate area
    area0 = (xmax0 - xmin0) * (ymax0 - ymin0)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    # Calculate overlap
    x_overlap = max(0, min(xmax0, xmax1) - max(xmin0, xmin1))
    y_overlap = max(0, min(ymax0, ymax1) - max(ymin0, ymin1))
    overlap = x_overlap * y_overlap
    return overlap / max(area0, area1)

# Maps predicted bounding boxes to true bounding boxes
def map_bounding_boxes(preds, bboxes, threshold):
    idxs = []
    idx_set = set()
    # Map each predicted bounding box to an true bounding box
    for box0 in preds["boxes"]:
        # Store max overlap percentage
        idx = -1
        max_overlap = -1
        # Iterate over true bounding boxes that have not already been mapped
        for i, box1 in [(j, box) for (j, box) in enumerate(bboxes) if j not in idx_set]:
            overlap = calculate_box_overlap(box0, box1)
            # Set true bounding box with max overlap
            if overlap > max_overlap:
                idx = i
                max_overlap = overlap
        # Set true bounding box with max overlap to mapped
        if max_overlap <= threshold:
            idx = -1
        idx_set.add(idx)
        idxs.append(idx)
    return idxs

def main():
    # Parse options
    opts, args = getopt.getopt(sys.argv[1:], "", ["show-ui"])
    show_ui = False
    for o, _ in opts:
        if o == "--show-ui":
            show_ui = True

    # Define model and image transforms
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load('webapp/faster_rcnn_model.pt', map_location=torch.device('cpu')))
    model = model.eval()
    transform = data_transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    # Get true bounding boxes and labels
    data_dir = '../data'
    image_dir = 'bounding_images'
    df = pd.read_csv(join(data_dir, 'classifier_input.csv'))
    images = df['Image'].values.tolist()
    bboxes = [parse_bounding_box(bboxes_string) for bboxes_string in 
        df['BoundingBoxes'].values.tolist()]
    labels = [parse_labels(labels_string) for labels_string in 
        df['Labels'].values.tolist()]

    model_labels = []
    # Iterate over images
    for img_idx, image in enumerate(images):
        print(image)
        img = Image.open(join(data_dir, image_dir, image)).convert('RGB')
        frame = None
        if show_ui:
            frame = cv2.cvtColor(asarray(img), cv2.COLOR_RGB2BGR)
        img = [data_transform(img)]
        preds = model(img)[0]
        l = preds["scores"].shape[0]
        preds["boxes"] = [[x.item() for x in boxes] for boxes in preds["boxes"]]
        # Contains indexs of generated bounding boxes without duplicates
        no_duplicate_idxs = []
        # Iterate over predicted bounding boxes and labels
        for i in range(l):
            # Check that current predicted bounding box does not overlap with any previous predicted bounding boxes by a threshold
            if map_bounding_boxes({"boxes": [preds["boxes"][i]]}, preds["boxes"][:i], 0.7)[0] == -1:
                # Get bounding box and label
                no_duplicate_idxs.append(i)

        # Map predicted bounding boxes to true bounding boxes
        bbox_idxs = map_bounding_boxes({"boxes": [preds["boxes"][idx] for idx in no_duplicate_idxs]}, bboxes[img_idx], 0.7)
        for i in range(len(no_duplicate_idxs)):
            if bbox_idxs[i] != -1:
                bbox_idx = bbox_idxs[i]
                pred_idx = no_duplicate_idxs[i]
                # Compute classification image file name given bounding image file name and bounding box index
                class_image = "%s-%s.png" % (image[:len(image)-4], bbox_idx)
                # Compute label
                class_num = preds["labels"][pred_idx].item()
                label = inverse_label_map[class_num]
                # Store to list
                model_labels.append((class_image, label))
                if show_ui:
                    # Draw computed bounding boxes
                    xmin, ymin, xmax, ymax = [int(x) for x in preds["boxes"][pred_idx]]
                    color = color_map[class_num]
                    cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)

                    # Draw true bounding boxes
                    bxmin, bymin, bxmax, bymax = [int(x) for x in bboxes[img_idx][bbox_idx]]
                    blabel = inverse_label_map[labels[img_idx][bbox_idx]]
                    cv2.putText(frame, blabel, (bxmin, bymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                    cv2.rectangle(frame, (bxmin, bymin), (bxmax, bymax), (0, 0, 0), 1)
        if show_ui:
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
    if show_ui:
        cv2.destroyAllWindows()
    df = pd.DataFrame(sorted(model_labels), columns=['Image', 'Label'])
    df.to_csv(join(data_dir, 'analysis', 'model_image_labels.csv'), index=False)
    

if __name__ == '__main__':
    main()