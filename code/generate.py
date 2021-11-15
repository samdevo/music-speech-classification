import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import random
import shutil

data_dir = "../data"

if (os.path.exists(os.path.join(data_dir, "train/music"))):
    shutil.rmtree("train/music")
if (os.path.exists(os.path.join(data_dir, "train/speech"))):
    shutil.rmtree("train/speech")
if (os.path.exists(os.path.join(data_dir, "test/music"))):
    shutil.rmtree("test/music")
if (os.path.exists(os.path.join(data_dir, "test/speech"))):
    shutil.rmtree("test/speech")

os.mkdir(os.path.join(data_dir, "train/music"))
os.mkdir(os.path.join(data_dir, "train/speech"))
os.mkdir(os.path.join(data_dir, "test/music"))
os.mkdir(os.path.join(data_dir, "test/speech"))


