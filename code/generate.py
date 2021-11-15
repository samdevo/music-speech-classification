import numpy as np
import os
import matplotlib.pyplot as plt
import random
import shutil
import librosa

random.seed(100)

data_dir = "../data"
train_wavs = 40

def clear_datadirs():

    if (os.path.exists(os.path.join(data_dir, "train"))):
        shutil.rmtree(os.path.join(data_dir, "train"))
    if (os.path.exists(os.path.join(data_dir, "test"))):
        shutil.rmtree(os.path.join(data_dir, "test"))

    os.mkdir(os.path.join(data_dir, "train"))
    os.mkdir(os.path.join(data_dir, "test"))
    os.mkdir(os.path.join(data_dir, "train/music"))
    os.mkdir(os.path.join(data_dir, "train/speech"))
    os.mkdir(os.path.join(data_dir, "test/music"))
    os.mkdir(os.path.join(data_dir, "test/speech"))

# takes wavs from ../data/WAVS_PATH and puts this data in the train and test folders
def create_spectrograms(wavs_path):
    wavs = os.listdir(os.path.join(data_dir, wavs_path + "_wav"))
    random.shuffle(wavs)
    for i, wav_file in enumerate(wavs):
        y, sr = librosa.load(os.path.join(data_dir, wavs_path + "_wav", wav_file), duration=30)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        db_converted = librosa.power_to_db(spectrogram, ref=np.max)[:, :1280]
        mini_grams = np.hsplit(db_converted, 10)
        train_or_test = "train" if i < train_wavs else "test"
        for j,gram in enumerate(mini_grams):
            # im = Image.fromarray(gram)
            filename = wav_file.replace(".wav", "") + "_" + str(j) + ".png"
            plt.imsave(os.path.join(data_dir, train_or_test, wavs_path, filename), gram)

        
clear_datadirs()
create_spectrograms("speech")
create_spectrograms("music")





