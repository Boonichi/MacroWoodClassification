import os
import pandas as pd
from pathlib import Path
import argparse
import json
import shutil
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv2 import imread

DATA_DIR = "./datasets/"
BRAZIL_WOOD_DIR = DATA_DIR + "brazillian_dataset/"
MENDELEY_WOOD_DIR = DATA_DIR + "mendeley_dataset/"
WOOD_AUTH_DIR = DATA_DIR + "Wood_AUTH/"
FSD_M_DIR = DATA_DIR + "FSD_M/"
OUTPUT_DIR = "./dataset/"
NUM_FOLDS = 2
SEED = 43

def identify_wood_name_and_label(data_dir : str, folder):
    if "brazillian"in data_dir:
        label, wood_name = folder.split("-")
        dataset_name = "brazillian"
    elif "Wood_AUTH" in data_dir:
        label, wood_name = folder.split(".")    
        dataset_name = "Wood_Auth"        
    elif "mendeley" in data_dir:
        wood_name = folder
        label = None
        dataset_name = "mendeley"
    elif "FSD_M" in data_dir:
        label, wood_name = folder.split(".")[0], folder.split(".")[1]
        dataset_name = "Wood_Auth"

    if wood_name is not None and wood_name[0] == "_": wood_name = wood_name[1:]
    return dataset_name, wood_name, label

def read_folder_dataset(df : list,data_dirs : str):

    # Create csv file include image path and its label
    for data_dir in data_dirs:
        for folder in os.listdir(data_dir):
            if (folder != ".DS_Store"):
                dataset_name, wood_name, label = identify_wood_name_and_label(data_dir, folder)
                for image in os.listdir(os.path.join(data_dir, folder)):
                    if (image.endswith(".jpg") or image.endswith(".png")) or image.endswith(".bmp"):
                        imgpath = os.path.join(data_dir, folder, image)
                        sample = {}
                        sample["imgpath"] = imgpath
                        sample["label"] = label
                        sample["wood_name"] = wood_name
                        sample["dataset"] = dataset_name

                        df.append(sample)

    return df

def prepare_data():
    df = []
    df = read_folder_dataset(df,[BRAZIL_WOOD_DIR, MENDELEY_WOOD_DIR, WOOD_AUTH_DIR, FSD_M_DIR])
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(os.path.join(OUTPUT_DIR, "full_dataset.csv"))

    # Label encoder wood_name feature
    LE = LabelEncoder()
    df["label"] = LE.fit_transform(df["wood_name"])

    # Name Mapping
    LE_name_mapping = {i: l for i, l in enumerate(LE.classes_)}

    f = open(os.path.join(OUTPUT_DIR,"name_mapping.json"), "w")
    json.dump(LE_name_mapping, f)
    f.close()

    # Split train/set
    
    X_train, X_test, Y_train, Y_test = train_test_split(df,df["label"], stratify=df["label"], test_size = 0.2)

    df_train = X_train.reset_index()
    df_test = X_test.reset_index()
    
    # Stratifiled K Folds (4 folds)
    skf = StratifiedKFold(n_splits = NUM_FOLDS, shuffle= True, random_state=SEED)
    df_train["fold"] = -1

    for i, (train_index, val_index) in enumerate(skf.split(df_train, df_train["label"])):
        df_train.loc[val_index, "fold"] = i

    print(df_train.groupby(['fold', 'label'])['imgpath'].size())

    df_train.to_csv(os.path.join(OUTPUT_DIR, "train_dataset.csv"))
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test_dataset.csv"))
    
    print(len(df_train))

    # Create Train/Test Folds
    for fold in range(NUM_FOLDS):
        fold_path = f"{OUTPUT_DIR}fold_{fold}/"
        train_path = f"{fold_path}train/"
        val_path = f"{fold_path}val/"
        
        os.makedirs(fold_path, exist_ok= True)
        os.makedirs(train_path, exist_ok= True)
        os.makedirs(val_path, exist_ok= True)

        train_df = df_train[df_train["fold"] != fold].reset_index(drop = True)
        val_df = df_train[df_train["fold"] == fold].reset_index(drop = True)

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
 