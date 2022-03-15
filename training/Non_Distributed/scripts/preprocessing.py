"""Author: Rishav Sapahia."""
import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed
import numpy as np
import cv2
ImageFile.Load_Truncated_Images = True


def trim(image_path):
    "https://codereview.stackexchange.com/a/132933/176687" 
    img2 = Image.open(image_path)
    img2 = np.array(img2)
    img_gray1 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    mask = img_gray1 > 20
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)+1
    cropped = img2[x0:x1, y0:y1]
    im =  Image.fromarray(cropped)
    return im


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = trim(image_path) 
    #img = Image.open(image_path)
    img = img.resize(
         (resize[1], resize[0]), resample=Image.LANCZOS
    )
    img.save(outpath)


path_dir = '/Volumes/My Book/Total_Dataset/OP'
output_folder = '/Volumes/My Book/Resized_Dataset/'
images = glob.glob(os.path.join(path_dir, '*.png'))


Parallel(n_jobs=28)(
     delayed(resize_image)(
         i,
         output_folder,
         (224, 224)
     )for i in tqdm(images)

 )
