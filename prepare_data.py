import os
import pandas as pd
from pathlib import Path
import argparse
import shutil

from sklearn.model_selection import StratifiedKFold, train_test_split

from cv2 import imread

DATA_DIR = "./wood_dataset/"
OUTPUT_DIR = "./dataset/"
NUM_FOLDS = 4
SEED = 43

def prepare_data():
    df = []

    # Create csv file include image path and its label
    for folder in os.listdir(DATA_DIR):
        if (folder != ".DS_Store"):
            label, wood_name = folder.split("-")

            for image in os.listdir(os.path.join(DATA_DIR, folder)):
                if (image.endswith("")):
                    imgpath = os.path.join(DATA_DIR, folder, image)
                    sample = {}
                    sample["imgpath"] = imgpath
                    sample["label"] = label
                    sample["wood_name"] = wood_name
                
                    df.append(sample)

    df = pd.DataFrame(df)

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
 