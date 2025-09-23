import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import soundfile as sf
import kaggle
import sys
import yaml

def main():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("iamhungundji/dysarthria-detection", path = None, unzip=True)

    os.makedirs("torgo/0", exist_ok=True)
    os.makedirs("torgo/1", exisit_ok=True)


    
    

