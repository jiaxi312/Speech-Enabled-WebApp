import tkinter
from tkinter import filedialog
import time
import pyaudio
import os.path
import numpy as np
import scipy.signal
import scipy.misc
import librosa
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import models
from PIL import Image
from PIL import ImageTk

# Global pyaudio stuff
RATE = 16000
BUFFER_SECONDS = 1.3

# Model paths
DAVENET_MODEL_PATH = './trained_models/davenet_vgg16_MISA_1024_pretrained/'
AUDIO_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'audio_model.pth')
IMAGE_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'image_model.pth')

class GuiApp(tkinter.Frame):
    def __init__(self, q, master=None, update_interval=200):
        # tkinter stuff
        tkinter.Frame.__init__(self, master)
        fram = tkinter.Frame(self)
        tkinter.Button(fram, text="Open File", command=self.open).pack(side=tkinter.LEFT)
        fram.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        self.update_interval = update_interval
        self.la = tkinter.Label(self)
        self.la.pack()
        self.pack()

        # signal processing stuff
        self.pa = pyaudio.PyAudio()
        self.coeff = 0.97
        self.window_size = 0.025
        self.window_stride =0.01
        self.hop_length = int(RATE * self.window_stride)
        self.n_fft = int(RATE * self.window_size)
        self.win_length = int(RATE * self.window_size)
        self.window = scipy.signal.hamming
        self.mel_basis = librosa.filters.mel(RATE, self.n_fft, n_mels=40, fmin=20)
        self.audio_buffer = np.zeros(int(RATE * BUFFER_SECONDS))

        # pytorch stuff
        self.audio_model, self.image_model = models.DAVEnet_model_loader(AUDIO_MODEL_PATH, IMAGE_MODEL_PATH)
        self.audio_model.eval()
        self.image_model.eval()
        self.image_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        self.matchmap_thresh = 1.3
        self.q = q

        # Start listening
        self.updater_id = None
        self.open_mic_stream()
        self.after(10, self.update_audio_buffer)

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

    def open_mic_stream(self):
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        INPUT_BLOCK_TIME = 0.01
        INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
        device_index = self.find_input_device()
        self.stream = self.pa.open(format = FORMAT,
                                channels = CHANNELS,
                                rate = RATE,
                                input = True,
                                input_device_index = device_index,
                                frames_per_buffer = INPUT_FRAMES_PER_BLOCK)
        self.stream.start_stream()

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

    def open(self):
        if self.updater_id is not None:
            self.after_cancel(updater_id)
            self.updater_id = None
        filename = filedialog.askopenfilename()
        if filename != "":
            self.base_image = Image.open(filename)
        self.change_image()
        self.update_image_features()
        self.updater_id = self.after(self.update_interval, self.update_matchmap)

    def change_image(self):
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

    def update_image_features(self):
        with torch.no_grad():
            self.image_model.eval()
            image_for_input = self.base_image.convert('RGB')
            self.full_W, self.full_H = image_for_input.size
            image_transformed = self.image_transform(image_for_input).unsqueeze(0)
            image_feature_map = self.image_model(image_transformed).squeeze(0)
            emb_dim = image_feature_map.size(0)
            self.output_H = image_feature_map.size(1)
            self.output_W = image_feature_map.size(2)
            self.image_features = image_feature_map.view(emb_dim, self.output_H*self.output_W)

    def update_matchmap(self):
        print('Updating matchmap')
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
            self.render_overlay(full_heatmap)
        self.after(self.update_interval, self.update_matchmap)

    def update_audio_buffer(self):
        try:
            in_data = self.stream.read(INPUT_FRAMES_PER_BLOCK)
            new_samples = np.fromstring(in_data, dtype=np.float32)
            N = new_samples.shape[0]
            self.audio_buffer[0:N] = new_samples
            self.audio_buffer = np.roll(self.audio_buffer, -1*N)
            print('Updated buffer with %d new samples' % N)
        except:
            print('Found no new samples')
            pass
        finally:
            self.after(10, self.check_queue)
        

def listen(q):
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    INPUT_BLOCK_TIME = 0.01
    INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

    print('Opening stream...')
    stream = pa.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

    while True:
        print('Reading from stream')
        q.put(stream.read(INPUT_FRAMES_PER_BLOCK))

if __name__ == '__main__':
    gui = GuiApp()
    gui.mainloop()