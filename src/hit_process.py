from os import listdir, mkdir
from os.path import isfile, join, isdir
from PIL import Image
from json import loads
from math import ceil
from random import sample
import pandas as pd

# Create text file with bounding box input image urls on S3 bucket
def create_bounding_image_urls():
    # Get all images in folder
    data_dir = '../data'
    path = join(data_dir, 'bounding_images')
    files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    f = open(join(data_dir, 'bounding_images.txt'),"w") 
    # Write images to text file and prepend with S3 bucket url
    for file in files:
        f.write('https://wym-mask-images.s3.amazonaws.com/' + file + '\n')

# Create input CSVs for the bounding box HIT task
def create_bounding_hit_inputs():
    # Read all image urls from bounding_images.txt
    data_dir = '../data'
    f = open(join(data_dir, 'bounding_images.txt'))
    files = sorted([url for url in f.readlines()])
    # Convert to CSVs
    df = pd.DataFrame(files[:500], columns=['image_url'])
    df.to_csv(join(data_dir, 'bounding_hit_input0.csv'), index=False)
    df = pd.DataFrame(files[500:], columns=['image_url'])
    df.to_csv(join(data_dir, 'bounding_hit_input1.csv'), index=False)
    
# Crop bounding box images based on bounding box HIT output and save cropped images
def crop_images():
    # Read bounding box HIT output CSV
    data_dir = '../data'
    df = pd.read_csv(join(data_dir, 'bounding_hit_output.csv'))
    bound_dir = 'bounding_images'
    class_dir = 'classification_images'
    # Create directory to store classification images
    if not isdir(join(data_dir, class_dir)):
        mkdir(join(data_dir, class_dir))
    # Iterate over CSV
    for f, bbox in df[['Input.image_url', 'Answer.annotatedResult.boundingBoxes']].values:
        f = f.replace('https://wym-mask-images.s3.amazonaws.com/', '')
        # Parse bounding box string
        l = loads(bbox)
        # Crop image for each bounding box and save to classification directory
        for i, bbox in enumerate(l):
            xmin = bbox['left']
            ymin = bbox['top']
            xmax = bbox['width'] + xmin
            ymax = bbox['height'] + ymin
            im=Image.open(join(data_dir, bound_dir, f))
            im=im.crop((xmin, ymin, xmax, ymax))
            crop_name = "%s-%s.png" % (f[:len(f)-4], i)
            im.save(join(data_dir, class_dir, crop_name), "PNG")

# Create text file with classification input image urls on S3 bucket
def create_classification_image_urls():
    # Get all images in folder
    data_dir = '../data'
    path = join(data_dir, 'classification_images')
    files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    f = open(join(data_dir, 'classification_images.txt'),"w") 
    # Write images to text file and prepend with S3 bucket url
    for file in files:
        f.write('https://wym-mask-images.s3.amazonaws.com/crop/' + file + '\n')
            
# Create CSV file inputs for classification HIT given gold standard labels
def create_classification_hit_inputs():
    # Read gold standard image labeled and unlabeled images
    data_dir = '../data'
    unlabeled = pd.read_csv(join(data_dir, 'unlabeled_images.csv'))['image_url'].values.tolist()
    df = pd.read_csv(join(data_dir, 'gold_standard_image_labels.csv'))
    nwm = []
    wmc = []
    wmi = []
    # Add gold standard image URLs to corresponding lists based on label
    for name, group in df.groupby(['label']):
        if name == 'Wearing Mask Correctly':
            wmc = group['image_url'].values.tolist()
        elif name == 'Wearing Mask Incorrectly':
            wmi = group['image_url'].values.tolist()
        elif name == 'Not Wearing Mask':
            nwm = group['image_url'].values.tolist()
        else:
            raise Exception('Unexpected label %s' % name)
    # Randomly add images to unlabeled images to make length a multiple of 6 
    num = len(unlabeled)
    start = num // 6 * 6
    diff = 6 - num % 6
    unlabeled += [unlabeled[i] for i in sample(range(start), diff)]
    l = []
    # Add rows of 6 unlabeled images and 3 gold standard images to a list
    for i in range(len(unlabeled) // 6):
        s = i * 6
        l.append((
            unlabeled[s], unlabeled[s+1], unlabeled[s+2], 
            unlabeled[s+3], unlabeled[s+4], unlabeled[s+5], 
            wmc[i % len(wmc)], wmi[i % len(wmi)], nwm[i % len(nwm)]
        ))
    # Convert to CSVs
    df = pd.DataFrame(l[:500], columns=['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'wmc_qc', 'wmi_qc', 'nwm_qc'])
    df.to_csv(join(data_dir, 'classification_hit_input0.csv'), index=False)
    df = pd.DataFrame(l[500:], columns=['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'wmc_qc', 'wmi_qc', 'nwm_qc'])
    df.to_csv(join(data_dir, 'classification_hit_input1.csv'), index=False)

def main():
    # Create text file with bounding box input image urls on S3 bucket
    #create_bounding_image_urls()
    # Create input CSVs for the bounding box HIT task
    #create_bounding_hit_inputs()
    # Crop bounding box images based on bounding box HIT output and save cropped images
    #crop_images()
    # Create text file with classification input image urls on S3 bucket
    #create_classification_image_urls()
    # Create CSV file inputs for classification HIT given gold standard labels
    #create_classification_hit_inputs()


if __name__ == '__main__':
    main()