import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import PIL.Image
from PIL import Image
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import pandas as pd
import math
import random

def features_random(node,model,random_number):
    mean = np.load("image_mean.npy")
    transform = transforms.Compose([
    transforms.Resize((256,256), PIL.Image.Resampling.BILINEAR),
    lambda x: np.array(x),
    lambda x: np.subtract(x[:,:,[2, 1, 0]], mean), #Subtract average mean from image (opposite order channels)
    lambda x: x[15:242, 15:242], #Center crop
    transforms.ToTensor()
    ])

    feature_extractor = create_feature_extractor(model, return_nodes=[node])
    image=[]
    activtiy=[]

    y =0
    while y <200:
        im = ("im")+str(random_number[y])+(".png")
        image.append(im)
        img = Image.open(im)
        image_array = np.array(img)
        if image_array.ndim < 3:
            image_array = np.repeat(image_array[..., np.newaxis], 3, axis=2)
        img = Image.fromarray(np.uint8(image_array))
        input = transform(img)
        input = input.unsqueeze(0)  
        out = feature_extractor(input)
        try:
            tmp=torch.reshape(out[node],(-1,1)).detach().numpy()
            model_features = tmp.ravel()
            activtiy.append(model_features)
            y+=1
        except TypeError:
            y+=1
            continue
    return pd.DataFrame(np.transpose(activtiy), columns=image)