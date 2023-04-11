import os
import pandas as pd
from pathlib import Path
import argparse
import shutil

from sklearn.model_selection import StratifiedKFold

from cv2 import imread

DATA_DIR = "./wood_dataset/"
OUTPUT_DIR = "./dataset/"
NUM_FOLDS = 5
SEED = 43

def Calc_mean_std(df):
    for index, row in df.iterrows():
        image = imread(row["imgpath"])
        print(image)
    
    #return mean, std
def prepare_data():
    df = []

    # Create csv file include image path and its label
    for folder in os.listdir(DATA_DIR):
        if (folder != ".DS_Store"):
            label, wood_name = folder.split("-")

            for image in os.listdir(os.path.join(DATA_DIR, folder)):
                if (image.endswith(""))
                imgpath = os.path.join(DATA_DIR, folder, image)
                sample = {}
                sample["imgpath"] = imgpath
                sample["label"] = label
                sample["wood_name"] = wood_name
                
                df.append(sample)

    df = pd.DataFrame(df)
    #df.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"))


    # Stratifiled K Folds (5 folds)
    skf = StratifiedKFold(n_splits = NUM_FOLDS, shuffle= True, random_state=SEED)
    df["fold"] = -1

    for i, (train_index, val_index) in enumerate(skf.split(df, df["label"])):
        df.loc[val_index, "fold"] = i

    print(df.groupby(['fold', 'label'])['imgpath'].size())

    df.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"))
    
    mean, std = Calc_mean_std(df)
    print("Mean {} and standard deviation {} of dataset".format(mean, std))
    for fold in range(NUM_FOLDS):
        fold_path = f"{OUTPUT_DIR}fold_{fold}/"
        train_path = f"{fold_path}train/"
        val_path = f"{fold_path}val/"
        
        os.makedirs(fold_path, exist_ok= True)
        os.makedirs(train_path, exist_ok= True)
        os.makedirs(val_path, exist_ok= True)

        train_df = df[df["fold"] != fold].reset_index(drop = True)
        val_df = df[df["fold"] == fold].reset_index(drop = True)

        for index, row in train_df.iterrows():   
            src_dir = row["imgpath"]
            img_name = src_dir.split("/")[-1]
            wood_name = row["wood_name"]

            destination_dir = f"{train_path}{wood_name}/{img_name}"
            os.makedirs(destination_dir, exist_ok=True)
            
            shutil.copy(src_dir, destination_dir)
        
        for index, row in val_df.iterrows():   
            src_dir = row["imgpath"]
            img_name = src_dir.split("/")[-1]
            wood_name = row["wood_name"]

            destination_dir = f"{val_path}{wood_name}/{img_name}"
            os.makedirs(destination_dir, exist_ok=True)
            
            shutil.copy(src_dir, destination_dir)

if __name__ == "__main__":
    if OUTPUT_DIR:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    prepare_data()
 