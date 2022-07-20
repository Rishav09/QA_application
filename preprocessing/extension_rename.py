import glob
import os
path = '/home/ubuntu/EyePacs_Lenke_Dataset/*'
total_files = glob.glob(path)

for image in total_files:
    base = os.path.splitext(image)[0]
    os.rename(image,base+'.JPG')

