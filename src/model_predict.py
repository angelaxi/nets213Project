from torchvision import transforms
from PIL import Image
from os import listdir
from os.path import join, isdir, isfile
from train_classifier import get_model_instance_segmentation
import pandas as pd
import torch
import getopt
import sys

data_dir = '../data'
analysis_dir = 'analysis'

inverse_label_map = {
    1: "Wearing Mask Correctly",
    2: "Wearing Mask Incorrectly",
    3: "Not Wearing Mask"
}

def main():
    image_dir = join(data_dir, 'validation_images')
    # Parse options
    opts, args = getopt.getopt(sys.argv[1:], "i:")
    for o, a in opts:
        if o == "-i":
            if isdir(a):
                image_dir = a
            else:
                raise Exception('%s is not a directory' % a)
    # Define model and image transforms
    model = get_model_instance_segmentation(3)
    model.load_state_dict(torch.load('webapp/faster_rcnn_model.pt', map_location=torch.device('cpu')))
    model = model.eval()
    transform = data_transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    # List of model predictions
    l = []
    # Get all images in a directory
    images =  [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    # Iterate over images
    for image in images:
        print(image)
        # Open image and convert to tensor
        img = Image.open(join(image_dir, image)).convert('RGB')
        img = [data_transform(img)]
        preds = model(img)[0]
        # Save predictions
        boxes = [[x.item() for x in boxes] for boxes in preds["boxes"]]
        labels = [x.item() for x in preds["labels"]]
        scores = [x.item() for x in preds["scores"]]
        l.append((image, boxes, labels, scores))
    # Convert to CSV
    df = pd.DataFrame(sorted(l), columns=['Image', 'BoundingBoxes', 'Labels', 'Scores'])
    df.to_csv(join(data_dir, analysis_dir, 'model_image_preds.csv'), index=False)
    

if __name__ == '__main__':
    main()