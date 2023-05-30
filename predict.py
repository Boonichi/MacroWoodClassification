import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score

from timm.models import create_model

from torchvision import transforms
import torch

import cv2
import models.convnext

CKPT_DIR = "./checkpoints/"
os.makedirs('./results/', exist_ok=True)

models = []
device = torch.device("mps")
nb_classes = 75

class Model(object):
    def __init__(self, model = "resnet152d", prefix = "", ckpt_path = None, size = 224):
        self.model = create_model(
            model,
            pretrained = False,
            num_classes = nb_classes,
        )
        self.tfsm = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0,0,0], std = [1,1,1])])
        

        self.model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        self.model.to(device)
        self.model.eval()

    def preprocess(self, input):
        output = self.tfsm(input).to(device)
        output = output.unsqueeze(0)
        return output
    def predict(self, input):
        input = self.preprocess(input)
        with torch.no_grad():
            output = self.model(input)
            output = output.softmax(1).to("cpu").numpy()

        return output
    
checkpoints = [
    CKPT_DIR + "resnet152d/checkpoint-best.pth",
    CKPT_DIR + "convnext_large/checkpoint-best.pth",
    CKPT_DIR + "swin_large_patch4_window7_224_in22k/checkpoint-best.pth"
]

names = ["resnet152d", "convnext_large", "swin_large_patch4_window7_224_in22k"]

print("Loading Model....")
for cp, name in zip(checkpoints, names):
    models.append(Model(model = name, prefix = "", ckpt_path = cp))


src = "./dataset/"
df = pd.read_csv(src + "test_dataset.csv")


for index, model in enumerate(models):
    print(model)
    outputs = []
    actuals = []

    for index, row in tqdm(df.iterrows()):
        imgpath = row["imgpath"]
        label = row["label"]
        woodname = row["wood_name"]
        wood_img = Image.open(imgpath)

        output = model.predict(wood_img)

        output = np.argmax(output) + 1
        #if output != label:
        #    print(imgpath)
        outputs.append(output)
        actuals.append(label)
    acc = accuracy_score(outputs, actuals)
    print(acc)

