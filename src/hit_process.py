from os import listdir, mkdir
from os.path import isfile, join, isdir
from PIL import Image
from json import loads
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

# Create input CSVs for the bounding box hit task
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
    
# Crop bounding box images based on bounding box hit output and save cropped images
def crop_images():
    # Read bounding box hit output CSV
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

def main():
    # Create text file with bounding box input image urls on S3 bucket
    create_bounding_image_urls()
    # Create input CSVs for the bounding box hit task
    create_bounding_hit_inputs()
    # Crop bounding box images based on bounding box hit output and save cropped images
    crop_images()


if __name__ == '__main__':
    main()