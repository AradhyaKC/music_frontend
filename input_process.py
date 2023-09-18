import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import math

def process(filename):
    SAMPLE_RATE=22050
    TRACK_DURATION=30
    SAMPLES_PER_TRACK=SAMPLE_RATE*TRACK_DURATION
    music=open(filename,'rb')

    df={"mfcc":[],"labels":[]}
    num_samples_per_segment=int(SAMPLES_PER_TRACK/10)
    expected_n_mfcc_vectors_per_segment=math.ceil(num_samples_per_segment/512)

    signal,sr=librosa.load(music,sr=SAMPLE_RATE)

    #process segments for mfcc extraction and storing
    for s in range(10):
        start_sample=num_samples_per_segment * s
        finish_sample=start_sample+num_samples_per_segment
        mfcc=librosa.feature.mfcc(y=signal[start_sample:finish_sample],sr=sr,n_fft=2048,n_mfcc=13,hop_length=512)
        mfcc=mfcc.T

        #storing mfcc if the segment has expected length
        if len(mfcc)==expected_n_mfcc_vectors_per_segment:
            df['mfcc'].append(mfcc.tolist())
            df['labels'].append(1)

    M=np.array(df['mfcc'])
    n=np.array(df['labels'])

    M_train,M_test,n_train,n_test=train_test_split(M,n,test_size=0.25)

    return M_test