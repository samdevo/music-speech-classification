import numpy as np
import os
import matplotlib.pyplot as plt
import random
import shutil
import librosa
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D



random.seed(100)

data_dir = "../data"
train_wavs = 40

# empties train and test folders to be refilled
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
        # find spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        db_converted = librosa.power_to_db(spectrogram, ref=np.max)[:, :1280]
        # split image into 10 128x128 squares
        mini_grams = np.hsplit(db_converted, 10)
        # decide proper destination for images
        train_or_test = "train" if i < train_wavs else "test"
        for j,gram in enumerate(mini_grams):
            # save 'image' in its path
            filename = wav_file.replace(".wav", "") + "_" + str(j) + ".png"
            plt.imsave(os.path.join(data_dir, train_or_test, wavs_path, filename), gram)

    
def reload_data():
    clear_datadirs()
    create_spectrograms("speech")
    create_spectrograms("music")

def create_generators():
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, "train"),
            color_mode='grayscale',
            target_size=(128, 128),
            batch_size=16,
            class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
            os.path.join(data_dir, "test"),
            color_mode='grayscale',
            target_size=(128, 128),
            batch_size=16,
            class_mode='binary')
    return train_generator, validation_generator

def create_model():
    model = Sequential()

    model.add(Conv2D(8, kernel_size=3, activation='relu', input_shape=(128,128,1)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model






if __name__ == "__main__": 
    # reload_data()
    train_generator, validation_generator = create_generators()
    model = create_model()

    model.fit(train_generator, 
        epochs=10, 
        verbose=1,
        validation_data=validation_generator, 
        validation_steps=800)

    print("evaluating model: ")
    print(model.evaluate(validation_generator))





