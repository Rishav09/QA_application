"""Source: https://medium.com/analytics-vidhya/how-to-pick-the-optimal-image-size-for-training-convolution-neural-network-65702b880f05"""
"""Code to find the right size to crop the image"""



import pandas as pd
from pathlib import Path
import imagesize
from tqdm import tqdm
from joblib import Parallel,delayed
root = '/home/ubuntu/EyePacs_Lenke_Dataset/'
output_folder = '/home/ubuntu/QA_code/QA_application/preprocessing'
imgs = [img.name for img in Path(root).iterdir() if img.suffix ==".JPG"]

img_meta = {}

for f in imgs:
    img_meta[str(f)] = imagesize.get(root+f)

img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
img_meta_df[["Width", "Height"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
img_meta_df["Aspect Ratio"] = round(img_meta_df["Width"] / img_meta_df["Height"], 2)
print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
destination_file = output_folder + "image_meta.csv"
img_meta_df.to_csv(destination_file)

#imgs = [img.name for img in Path(root).iterdir() if img.suffix ==".JPG"]
#output_folder = '/home/ubuntu/QA_code/QA_application/preprocessing'
#Parallel(n_jobs=120)(
      # delayed(resize_image)(
       # root,
       # output_folder
       # )for i in tqdm(imgs)
       # )



