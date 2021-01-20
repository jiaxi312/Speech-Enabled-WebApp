#!/usr/bin/python

# open a microphone in pyAudio and listen for taps

import pyaudio
import struct
import math
import sys
import time
import os.path
import numpy as np
import scipy.signal
import scipy.misc
import librosa
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import models as models
import wave


#FORMAT = pyaudio.paInt16 
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000  
INPUT_BLOCK_TIME = 0.01
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

DAVENET_MODEL_PATH = './trained_models/davenet_vgg16_MISA_1024_pretrained/'
AUDIO_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'audio_model.pth')
IMAGE_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'image_model.pth')
MATCHMAP_THRESH = 10

BUFFER_SECONDS=1.3

def get_MatchMapper():
    audio_model, image_model = models.DAVEnet_model_loader(AUDIO_MODEL_PATH, IMAGE_MODEL_PATH)
    return MatchMapper(audio_model, image_model)

class MatchMapper(object):
    def __init__(self, audio_model, image_model):
        self.audio_model = audio_model
        self.image_model = image_model
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.errorcount = 0
        self.audio_buffer = np.zeros(int(RATE * BUFFER_SECONDS))
        self.image_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        # signal processing stuff
        self.coeff = 0.97
        self.window_size = 0.025
        self.window_stride =0.01
        self.hop_length = int(RATE * self.window_stride)
        self.n_fft = int(RATE * self.window_size)
        self.win_length = int(RATE * self.window_size)
        self.window = scipy.signal.hamming
        self.mel_basis = librosa.filters.mel(RATE, self.n_fft, n_mels=40, fmin=20)

    # Compute based on input audio, extract features from audio.
    # y is the audio data
    def spectrogram(self, y):
        # y = self.audio_buffer
        y = y - y.mean()
        y = np.append(y[0],y[1:]-self.coeff*y[:-1])
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window)
        spec = np.abs(stft)**2
        melspec = np.dot(self.mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
        logspec = torch.FloatTensor(logspec)
        return logspec

    def demo(self, image_file, wf):
        with torch.no_grad():
            self.image_model.eval()
            self.audio_model.eval()
            try:
                image_fullsize = Image.open(image_file).convert('RGB')
            except IOError:
                print("Couldn't read the image file, try again!")
                return
            full_W, full_H = image_fullsize.size
            image_input = self.image_transform(image_fullsize).unsqueeze(0)
            image_feature_map = self.image_model(image_input).squeeze(0)

            E = image_feature_map.size(0)
            H = image_feature_map.size(1)
            W = image_feature_map.size(2)
            image_features = image_feature_map.view(E, H*W)

            #figure, ax = plt.subplots(1, 1)
            figure = plt.figure(figsize=(7, 7))
            spec = self.spectrogram()
            # background = plt.imshow(image_fullsize)
            # overlay = plt.imshow(np.ones((full_H, full_W)), cmap='gist_gray_r', alpha=0.5, vmin=0, vmax=255)

            print(image_fullsize.size)
            print(image_feature_map.size())

            # No use
            def stream_callback(in_data, frame_count=None, time_info=None, status=None):
                new_samples = np.fromstring(in_data, dtype=np.float32)
                N = new_samples.shape[0]
                self.audio_buffer[0:N] = new_samples
                self.audio_buffer = np.roll(self.audio_buffer, -1*N)
                # return (None, pyaudio.paContinue)

            def update_plot(frame):
                print('update plot')
                y, rate = librosa.load(wf, 16000)
                spec = self.spectrogram(y)
                audio_output = self.audio_model(spec.unsqueeze(0).unsqueeze(0)).squeeze(0)
                # audio_output is (1024, T)
                T = audio_output.size(1)
                heatmap = torch.mm(audio_output.t(), image_features).squeeze().view(T, H, W).max(dim=0)[0].numpy()
                h_mean = heatmap.mean()
                h_min = heatmap.min()
                h_max = heatmap.max()
                #print('min = %.2f, mean = %.2f, max = %.2f' % (h_min, h_mean, h_max))
                heatmap = np.where(heatmap >= MATCHMAP_THRESH, 0, 1)
                #heatmap = 1. - (heatmap >= MATCHMAP_THRESH).float()
                #full_heatmap = scipy.misc.imresize(heatmap, (full_H, full_W), mode='F', interp='bilinear')
                full_heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((full_W, full_H), resample=Image.BILINEAR))
                # Merge full_heatmap with image
                return overlay

            # animation = FuncAnimation(figure, update_plot, interval=200)

            # self.open_mic_stream(stream_callback, wf)
            # self.stream = self.pa.open(format=self.pa.get_format_from_width(wf.getsampwidth()),
            #       channels=wf.getnchannels(),
            #       rate=wf.getframerate(),
            #       output=True)

            plt.show()
            print('plot show')

            # if self.stream.is_active():
            #     plt.show()

            data = wf.readframes(1024)

            # play stream (3)
            while len(data) > 0:
                # self.stream.write(data)
                data = wf.readframes(1024)
                self.audio_buffer = data

            # stop stream (4)
            # self.stream.stop_stream()
            self.stream.close()

            self.stop()
            return

    def stop(self):
        self.stream.close()

    def find_input_device(self):
        device_index = None            
        for i in range( self.pa.get_device_count() ):     
            devinfo = self.pa.get_device_info_by_index(i)   
            print( "Device %d: %s"%(i,devinfo["name"]) )

            for keyword in ["mic","input"]:
                if keyword in devinfo["name"].lower():
                    print( "Found an input: device %d - %s"%(i,devinfo["name"]) )
                    device_index = i
                    return device_index

        if device_index == None:
            print( "No preferred input found; using default input device." )

        return device_index

    def open_mic_stream(self, callback, wf):
        # self.stream = self.pa.open(format=self.pa.get_format_from_width(wf.getsampwidth()),
        #                         channels = wf.getnchannels,
        #                         rate=wf.getframerate(),
        #                         output=True,
        #                         frames_per_buffer = INPUT_FRAMES_PER_BLOCK,
                                # stream_callback = callback)
        self.stream = self.pa.open(format=self.pa.get_format_from_width(wf.getsampwidth()),
                  channels=wf.getnchannels(),
                  rate=wf.getframerate(),
                  output=True)
        data = wf.readframes(1024)

        # play stream (3)
        while len(data) > 0:
            self.stream.write(data)
            data = wf.readframes(1024)

        # stop stream (4)
        self.stream.stop_stream()
        self.stream.close()
        return self


if __name__ == "__main__":
    #if len(sys.argv) != 2:
    #    print("Run a DAVEnet in matchmap demo mode for an input image, using the microphone as input.\n\n"
    #        "Usage: %s filename.jpg" % sys.argv[0])
    #    sys.exit(-1)

    audio_model, image_model = models.DAVEnet_model_loader(AUDIO_MODEL_PATH, IMAGE_MODEL_PATH)

    mm = MatchMapper(audio_model, image_model)

    while True:
        input_str = input("Enter a path to an image file to load, an ls command, or ctrl-c to exit:")
        toks = input_str.split(' ')
        if toks[0] == 'ls':
            os.system(input_str)
        else:
            wf = wave.open('utterance_1890.wav')
            mm.demo(input_str, wf)
