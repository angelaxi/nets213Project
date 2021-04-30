from torchvision import transforms
from PIL import Image
from numpy import asarray
from os.path import join, isfile
from train_classifier import get_model_instance_segmentation
import pandas as pd
import torch
import cv2

data_dir = '../data'
analysis_dir = 'analysis'

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

def main():
    f = join(data_dir, analysis, 'model_performance.csv')
    df = None
    if isfile(f):
        df = pd.read_csv(df)
    else:
        df = DataFrame(columns=['Image', 'BoundingBoxes', 'Labels', ])
    
    

if __name__ == '__main__':
    main()