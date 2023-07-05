import librosa
import os
import random
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_features(y, sr=44100):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) #Chroma Spectogram
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    # chroma_vqt = librosa.feature.chroma_vqt(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr) # mfc coefficients
    rmse = librosa.feature.rms(y=y) #root mean square
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)#spectral centroid
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr) #spectral bandwidth
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness= librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr) #spectral roll off
    poly_features = librosa.feature.poly_features(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y) #zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y) #zero crossing rate
    fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr) #Chroma Spectogram

    return [chroma_stft,chroma_cqt,chroma_cens,melspectrogram,mfcc,rmse,spec_cent,spec_bw,spectral_contrast,
            spectral_flatness,rolloff,poly_features,tonnetz,zcr,fourier_tempogram]

def GETFILE(f):
    L = librosa.get_duration(filename=f) -20 #fist we calculate the duration of the input sound
    y, sr = librosa.load(f, sr=44100, duration=20, offset=random.randrange(int(L))) #then we sample 20 second from a random point in file
    y/=y.max() #then we normalize the input so all sounds have same volume
    k = get_features(y) #then we extract the features
    #then add the whole features into a string file so we can add them all together into feature vector
    to_append = f''
    for kk in k:
    # print(len(kk))
        for e in kk:
            # print(len(kk))
            # print(type(np.mean(e)))
            if(type(np.mean(e))==type(np.complex64(1+2j))):
                to_append += f' {np.mean(e.real)}'
                to_append += f' {np.mean(e.imag)}'
            else:
                to_append += f' {np.mean(e)}'
    feature_vectors=to_append.split()
    # print(len(to_append.split()))
    label1=int(f.split('/')[-2].split('_')[1])
    label2=int(f.split('/')[-3].split('_')[1])
    #finally turning feature_vector into numpy array
    feature_vectors = np.array(feature_vectors)
    return feature_vectors,label1,label2

# f="./D_0/I_4/ahang.mp3"

# GETFILE(f)

