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
import models

try:
    from Tkinter import *
    import tkFileDialog as filedialog
except ImportError:
    from tkinter import *
    from tkinter import filedialog

from PIL import Image
from PIL import ImageTk

DAVENET_MODEL_PATH = './trained_models/davenet_vgg16_MISA_1024_pretrained/'
AUDIO_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'audio_model.pth')
IMAGE_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'image_model.pth')

class ImageDisplayer(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.matchmapper = None
        self.master.title('DAVEnet Machmap Demo')

        fram = Frame(self)
        Button(fram, text="Open File", command=self.open).pack(side=LEFT)
        fram.pack(side=TOP, fill=BOTH)

        self.la = Label(self)
        self.la.pack()

        self.pack()

    def open(self):
        filename = filedialog.askopenfilename()
        if filename != "":
            self.base_image = Image.open(filename)
        self.chg_image()
        self.matchmapper.update_image()

    def chg_image(self):
        if self.base_image.mode == "1": # bitmap image
            self.display_image = ImageTk.BitmapImage(self.base_image, foreground="white")
        else:              # photo image
            self.display_image = ImageTk.PhotoImage(self.base_image)
        self.la.config(image=self.display_image, bg="#000000",
            width=self.display_image.width(), height=self.display_image.height())

    def render_overlay(self, overlay):
        background = self.base_image.copy()
        background.putalpha(overlay)
        #background.paste(overlay, (0, 0), overlay.convert('RGBA'))
        if background.mode == "1":
            self.display_image = ImageTk.BitmapImage(background, foreground="white")
        else:
            self.display_image = ImageTk.PhotoImage(background)
        self.la.config(image=self.display_image, bg="#000000",
            width=self.display_image.width(), height=self.display_image.height())
        #overlay = plt.imshow(np.ones((full_H, full_W)), cmap='gist_gray_r', alpha=0.5, vmin=0, vmax=255)

class MatchMapper():
    def __init__(self):
        self.displayer = None

        # Streaming audio stuff
        self.rate = 16000
        self.buffer_seconds = 1.3
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.input_block_time = 0.01
        self.input_frames_per_block = int(self.rate * self.input_block_time)

        self.audio_model, self.image_model = models.DAVEnet_model_loader(AUDIO_MODEL_PATH, IMAGE_MODEL_PATH)
        self.audio_model.eval()
        self.image_model.eval()
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.errorcount = 0
        self.audio_buffer = np.zeros(int(self.rate * self.buffer_seconds))
        self.image_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        self.image_features = None

        # signal processing stuff
        self.coeff = 0.97
        self.window_size = 0.025
        self.window_stride =0.01
        self.hop_length = int(self.rate * self.window_stride)
        self.n_fft = int(self.rate * self.window_size)
        self.win_length = int(self.rate * self.window_size)
        self.window = scipy.signal.hamming
        self.mel_basis = librosa.filters.mel(self.rate, self.n_fft, n_mels=40, fmin=20)

        self.update_interval = .2
        self.matchmap_thresh = 1.3

    def spectrogram(self):
        y = self.audio_buffer
        y = y - y.mean()
        y = np.append(y[0],y[1:]-self.coeff*y[:-1])
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window)
        spec = np.abs(stft)**2
        melspec = np.dot(self.mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
        logspec = torch.FloatTensor(logspec)
        return logspec

    def stop(self):
        if self.stream is not None:
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
        self.stream = self.pa.open(format = self.format,
                                channels = self.channels,
                                rate = self.rate,
                                input = True,
                                input_device_index = device_index,
                                frames_per_buffer = self.input_frames_per_block,
                                stream_callback = callback)
        self.stream.start_stream()
        return self

    def update_image(self):
        with torch.no_grad():
            self.image_model.eval()
            image_for_input = self.displayer.base_image.convert('RGB')
            self.full_W, self.full_H = image_for_input.size
            image_transformed = self.image_transform(image_for_input).unsqueeze(0)
            image_feature_map = self.image_model(image_transformed).squeeze(0)
            emb_dim = image_feature_map.size(0)
            self.output_H = image_feature_map.size(1)
            self.output_W = image_feature_map.size(2)
            self.image_features = image_feature_map.view(emb_dim, self.output_H*self.output_W)
            
    def update_overlay(self):
        if self.image_features is None:
            return
        with torch.no_grad():
            spec = self.spectrogram()
            audio_output = self.audio_model(spec.unsqueeze(0).unsqueeze(0)).squeeze(0)
            # audio_output is (1024, T)
            T = audio_output.size(1)
            heatmap = torch.mm(audio_output.t(), self.image_features).squeeze().view(T, self.output_H, self.output_W).max(dim=0)[0].numpy()
            h_mean = heatmap.mean()
            h_min = heatmap.min()
            h_max = heatmap.max()
            #print('min = %.2f, mean = %.2f, max = %.2f' % (h_min, h_mean, h_max))
            heatmap = np.where(heatmap >= self.matchmap_thresh, 0, 1)
            #heatmap = 1. - (heatmap >= MATCHMAP_THRESH).float()
            #full_heatmap = scipy.misc.imresize(heatmap, (full_H, full_W), mode='F', interp='bilinear')
            full_heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((self.full_W, self.full_H), resample=Image.BILINEAR)
            self.displayer.render_overlay(full_heatmap)

    def run(self):
        self.audio_model.eval()
        self.num_blocks_since_last_update = 0

        def stream_callback(in_data, frame_count, time_info, status):
            new_samples = np.fromstring(in_data, dtype=np.float32)
            N = new_samples.shape[0]
            self.audio_buffer[0:N] = new_samples
            self.audio_buffer = np.roll(self.audio_buffer, -1*N)
            return (None, pyaudio.paContinue)

        self.open_mic_stream(stream_callback)
            
        while True:
            self.update_overlay()
            #time.sleep(self.update_interval)


if __name__ == "__main__":
    displayer = ImageDisplayer()
    matchmapper = MatchMapper()
    displayer.matchmapper = matchmapper
    matchmapper.displayer = displayer
    displayer.after(matchmapper.update_interval, displayer.)
    displayer.mainloop()