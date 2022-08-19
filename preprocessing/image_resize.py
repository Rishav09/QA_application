"""Author: Rishav Sapahia."""
import os
import glob
from tqdm import tqdm
from PIL import Image,ImageFile
from joblib import Parallel,delayed

ImageFile.Load_Truncated_Images = True


def resize_image(image_path,output_folder,resize):
        base_name = os.path.basename(image_path)
        outpath = os.path.join(output_folder, base_name)
        img = Image.open(image_path)
        rgb_img = img.convert('RGB')
        rgb_img = rgb_img.resize(
         (resize[1], resize[0]), resample=Image.BILINEAR
         )
        rgb_img.save(outpath)


path_dir = '/home/ubuntu/Left_AI_Bias'
output_folder = '/home/ubuntu/Left_AI_Bias_448/'
images = glob.glob(os.path.join(path_dir, '*.JPG'))


Parallel(n_jobs=120)(
     delayed(resize_image)(
         i,
         output_folder,
         (448, 448)
     )for i in tqdm(images)
 )
