from PIL import Image
from os import mkdir
from os.path import join, isdir
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from json import loads
import pandas as pd
import torch
import torchvision

label_map = {
    "Wearing Mask Correctly": 1,
    "Wearing Mask Incorrectly": 2,
    "Not Wearing Mask": 3
}

# Parse bounding boxes
def parse_bounding_box(bboxes_string):
    bboxes = loads(bboxes_string)
    return [[
        bbox['left'], bbox['top'], bbox['width'] + bbox['left'], 
        bbox['height'] + bbox['top']] for bbox in bboxes]

def parse_labels(labels_string):
    labels = loads(labels_string)
    return [label_map[label] for label in labels]

class MaskDataset(object):
    # Load image labels
    def __init__(self, transforms):
        self.transforms = transforms
        df = pd.read_csv('../data/classifier_input.csv')
        self.images = df['Image'].values.tolist()
        self.bboxes = [parse_bounding_box(bboxes_string) for bboxes_string in 
            df['BoundingBoxes'].values.tolist()]
        self.labels = [parse_labels(labels_string) for labels_string in 
            df['Labels'].values.tolist()]

    # Load image with label
    def __getitem__(self, idx):
        image_dir = '../data/bounding_images'
        img = Image.open(join(image_dir, self.images[idx])).convert("RGB")
        # Apply image transformations
        if self.transforms is not None:
            img = self.transforms(img)

        return img, {
            'image_id': torch.tensor([idx]),
            'boxes': torch.as_tensor(self.bboxes[idx], dtype=torch.int64),
            'labels': torch.as_tensor(self.labels[idx], dtype=torch.int64)
        }

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))

# Load pretrained segmentation model
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    return model

def main():
    # Contruct dataset
    data_transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    dataset = MaskDataset(data_transform)
    dataset = MaskDataset(data_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=collate_fn)
    # Get pretrained model
    model = get_model_instance_segmentation(3)
    # Check if cuda is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Define parameters
    num_epochs = 25
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, 
                                momentum=0.9, weight_decay=0.0005)
    data_len = len(data_loader)
    for epoch in range(num_epochs):
        model.train()
        i = 0    
        epoch_loss = 0
        for images, boxlabels in data_loader:
            i += 1
            images = [image.to(device) for image in images]
            boxlabels = [{k: v.to(device) for k, v in t.items()} for t in boxlabels]
            loss_dict = model(images, boxlabels)
            losses = sum(loss for loss in loss_dict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
            print(f'Iteration: {i}/{data_len}, Loss: {losses}')
            epoch_loss += losses
            torch.cuda.empty_cache()
        print(epoch_loss)
    # save model
    models_dir = 'models'
    if not isdir(models_dir):
        mkdir(models_dir)
    torch.save(model.state_dict(),join(models_dir, 'faster_rcnn_model.pt'))

if __name__ == '__main__':
    main()