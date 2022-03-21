import time  # sinon pip install python-time
import wave
import os
import random
import string
import sounddevice as sd  # sinon pip install sounddevice
import scipy.io.wavfile as wav
from python_speech_features import mfcc  # sinon pip install python_speech_features==0.4
import numpy as np
import pandas as pd


def save_wav(data, rate, file_path: str):
    """Save the input as a .wave file

    :data: Sample to save
    :rate: Sampling rate
    :file_path: Destination uri
    :returns: TODO

    """
    with wave.open(file_path, mode='wb') as wb:
        wb.setnchannels(1)  # monaural
        wb.setsampwidth(2)  # 16bit=2byte
        wb.setframerate(rate)
        wb.writeframes(data.tobytes())  # Convert to byte string


# Digit Recognition

def rec(scaler, classifier):

    print("Attention, l'enregistrement commence dans :")
    (rate, sig) = wav.read("./Tools/beep-04.wav")
    sd.play(sig, rate)
    for i in range(0, 6):
        time.sleep(1)
        print(5-i)

    time.sleep(1)

    rate = 48000
    duration = 2

    print("Prononcer votre Digit : ")
    data = sd.rec(int(duration * rate), samplerate=rate, channels=1)
    sd.wait()

    data = data / data.max() * np.iinfo(np.int16).max
    data = data.astype(np.int16)

    save_wav(data, rate, './Rec/Rec_'+'Capture'+'_.wav')

    mfcc_feat = np.mean(mfcc(data, rate, numcep=12), axis=0)
    mfcc_feat = np.expand_dims(mfcc_feat, axis=0)
    pred = classifier.predict(scaler.transform(mfcc_feat))
    print('------------------')
    print('Digit (prédiction) : ', pred[0])
    print('------------------')
    return pred

# Digit Collection


def collection():
    df = pd.DataFrame
    name_generator = ''.join(random.sample(string.ascii_lowercase, 3))
    #chr(np.random.randint(500))+chr(np.random.randint(500))+chr(np.random.randint(500))

    print("Attention, l'enregistrement commence dans :")
    (rate, sig) = wav.read("Tools/beep-04.wav")
    sd.play(sig, rate)
    for sec in range(0, 6):
        time.sleep(1)
        print(5-sec)

    time.sleep(1)

    rate = 48000
    duration = 2

    for chiffre in range(0, 10):

        print("Prononcer le chiffre : "+str(chiffre))
        data = sd.rec(int(duration * rate), samplerate=rate, channels=1)
        sd.wait()

        data = data / data.max() * np.iinfo(np.int16).max
        data = data.astype(np.int16)

        save_wav(data, rate, './Rec/Rec_'+str(chiffre)+'_.wav')

        mfcc_feat = np.mean(mfcc(data, rate, numcep=12), axis=0)

        d = {'Fe1': mfcc_feat[0],
             'Fe2': mfcc_feat[1],
             'Fe3': mfcc_feat[2],
             'Fe4': mfcc_feat[3],
             'Fe5': mfcc_feat[4],
             'Fe6': mfcc_feat[5],
             'Fe7': mfcc_feat[6],
             'Fe8': mfcc_feat[7],
             'Fe9': mfcc_feat[8],
             'Fe10': mfcc_feat[9],
             'Fe11': mfcc_feat[10],
             'Fe12': mfcc_feat[11],
             'Target': chiffre,
             }

        if any(File.endswith(".csv") for File in os.listdir('./Dataset/')):
            df = pd.read_csv('./DataSet/'+os.listdir('./DataSet/')[0])
            df = df.append(d, ignore_index=True)
            df.to_csv('./DataSet/'+os.listdir('./DataSet/')[0], index=False)
        else:
            df = pd.DataFrame(d, index=['1'])
            df.to_csv('./DataSet/DataSet__'+name_generator+'__.csv', index=False)

    return df.head()
