"""Rishav Sapahia."""
import sys
sys.path.insert(1,'/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Application/Quality_scanner') # noqa
import os
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from import_packages.checkpoint import load_checkpoint
import torch.nn as nn
import random
from efficientnet_pytorch import EfficientNet

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)
# CuDA Determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 2

model_transfer = EfficientNet.from_pretrained('efficientnet-b2')
n_inputs = model_transfer._fc.in_features
model_transfer._fc = nn.Linear(n_inputs, 3) # noqa

model = load_checkpoint(checkpoint_path='/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Data_models/chocolate_feather/checkpoint_224.pt',model = model_transfer) # noqa

response = open("/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Application/Quality_scanner/gradio_UI/label_names.txt","r") # noqa
labels = [line.rstrip('\n') for line in response]


def predict(inp):
    """Function."""
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    inp = inp.resize(
         (224, 224), resample=Image.BILINEAR
     )
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        output = model_transfer(inp)
        output = torch.nn.functional.softmax(model_transfer(inp), dim=1).squeeze() # noqa
        output = output.detach().numpy()
    return {labels[i]: float(output[i]) for i in range(3)}


inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=3)
gr.Interface(fn=predict, inputs=inputs, outputs=outputs).launch()
