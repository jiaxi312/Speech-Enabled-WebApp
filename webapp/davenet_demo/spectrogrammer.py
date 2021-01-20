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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import models

#FORMAT = pyaudio.paInt16 
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000  
INPUT_BLOCK_TIME = 0.01
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

BUFFER_SECONDS=2

class Spectrogrammer(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.errorcount = 0
        self.audio_buffer = np.zeros(int(RATE * BUFFER_SECONDS))

        # signal processing stuff
        self.coeff = 0.97
        self.window_size = 0.010
        self.window_stride =0.002
        self.hop_length = int(RATE * self.window_stride)
        self.n_fft = 1024 #int(RATE * self.window_size)
        self.win_length = int(RATE * self.window_size)
        self.window = scipy.signal.hamming

    def spectrogram(self):
        y = self.audio_buffer
        y = y - y.mean()
        y = np.append(y[0],y[1:]-self.coeff*y[:-1])
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window)
        spec = np.abs(stft)**2
        logspec = librosa.power_to_db(spec, ref=np.max)
        logspec = np.flipud(logspec)
        return logspec

    def demo(self):
        figure = plt.figure(figsize=(14, 5))
        spec = self.spectrogram()
#        ax.set_xlim(0, self.audio_buffer.shape[0])
#        ax.set_ylim(-1, 1)

        plotted = plt.imshow(spec, cmap='gist_gray_r', vmin=-60, vmax=0)
        
        def stream_callback(in_data, frame_count, time_info, status):
            new_samples = np.fromstring(in_data, dtype=np.float32)
            N = new_samples.shape[0]
            self.audio_buffer[0:N] = new_samples
            self.audio_buffer = np.roll(self.audio_buffer, -1*N)
            return (None, pyaudio.paContinue)

        def update_plot(frame):
            spec = self.spectrogram()
            plotted.set_data(spec)
            plotted.autoscale()
            return plotted

        animation = FuncAnimation(figure, update_plot, interval=50)

        self.open_mic_stream(stream_callback)

        while self.stream.is_active():
            plt.show()

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

    def open_mic_stream(self, callback):
        device_index = self.find_input_device()
        self.stream = self.pa.open(  format = FORMAT,
                                channels = CHANNELS,
                                rate = RATE,
                                input = True,
                                input_device_index = device_index,
                                frames_per_buffer = INPUT_FRAMES_PER_BLOCK,
                                stream_callback = callback)
        self.stream.start_stream()
        return self


if __name__ == "__main__":
    s = Spectrogrammer()
    s.demo()
