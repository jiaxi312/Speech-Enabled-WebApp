import davenet_demo.models as models
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
import scipy.signal
import torch
import torchvision.transforms as transforms

from matplotlib.animation import FuncAnimation
from PIL import Image


matplotlib.use('Agg')

CHANNELS = 1
RATE = 16000  
INPUT_BLOCK_TIME = 0.01
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

DAVENET_MODEL_PATH = './webapp/davenet_demo/trained_models/davenet_vgg16_MISA_1024_pretrained/'
AUDIO_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'audio_model.pth')
IMAGE_MODEL_PATH = os.path.join(DAVENET_MODEL_PATH, 'image_model.pth')
# Large, nothing will light out.
MATCHMAP_THRESH = 10

BUFFER_SECONDS = 1.3

OVERLAP_SECONDS = 0.25


class MatchMapper(object):
    def __init__(self, audio_model, image_model):
        self.audio_model = audio_model
        self.image_model = image_model
        self.errorcount = 0
        self.audio_buffer = np.zeros(int(RATE * BUFFER_SECONDS))
        self.image_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        self.audio_buffer = np.zeros(int(RATE * BUFFER_SECONDS))

        # signal processing stuff
        self.coeff = 0.97
        self.window_size = 0.025
        self.window_stride =0.01
        self.hop_length = int(RATE * self.window_stride)
        self.n_fft = int(RATE * self.window_size)
        self.win_length = int(RATE * self.window_size)
        self.window = scipy.signal.hamming
        self.mel_basis = librosa.filters.mel(RATE, self.n_fft, n_mels=40, fmin=20)

    def spectrogram(self, y):
        y = y - y.mean()
        y = np.append(y[0],y[1:]-self.coeff*y[:-1])
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window)
        spec = np.abs(stft)**2
        melspec = np.dot(self.mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
        logspec = torch.FloatTensor(logspec)
        return logspec

    def gen(self, y):
        # add BUFFER_SECONDS * RATE seconds to y
        y = np.concatenate([np.zeros(int(BUFFER_SECONDS * RATE)), y])
        start, stop = 0, int(min(y.shape[0],BUFFER_SECONDS * RATE))
        count = 0

        while stop < y.shape[0]:
            count += 1
            yield y[start : stop]
            start, stop = start + int(OVERLAP_SECONDS * RATE), stop + int(OVERLAP_SECONDS * RATE)
            
        # Process the last slice of the audio if any
        if stop != y.shape[0]:
            count += 1
            yield y[start : y.shape[0]]
        print('frames: ' + str(count))

    
    def demo_with_animation(self, image_file, wf):
        with torch.no_grad():
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
            background = plt.imshow(image_fullsize)
            overlay = plt.imshow(np.ones((full_H, full_W)), cmap='gist_gray_r', alpha=0.5, vmin=0, vmax=255)
            plt.axis('off')

            y, rate = librosa.load(wf, sr=RATE)

            def update_plot(audio):
                spec = self.spectrogram(audio)
                audio_output = self.audio_model(spec.unsqueeze(0).unsqueeze(0)).squeeze(0)
                T = audio_output.size(1)
                heatmap = torch.mm(audio_output.t(), image_features).squeeze().view(T, H, W).max(dim=0)[0].numpy()
                h_mean = heatmap.mean()
                h_min = heatmap.min()
                h_max = heatmap.max()
                heatmap = np.where(heatmap >= MATCHMAP_THRESH, 0, 1)
                full_heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((full_W, full_H), resample=Image.BILINEAR))
                overlay.set_data(full_heatmap)
                return overlay
            
            anim = FuncAnimation(figure, update_plot, frames=self.gen(y),interval=int(OVERLAP_SECONDS * 1000))
            video_name = './webapp/static/files/video_animation.mp4'
            anim.save(video_name, writer = 'ffmpeg')
            output_name = './webapp/static/files/output.mp4'
            os.system(f'ffmpeg -i {video_name} -i {wf} -map 0:v -map 1:a -c:v copy -shortest {output_name}')
                        
    

if __name__ == "__main__":

    audio_model, image_model = models.DAVEnet_model_loader(AUDIO_MODEL_PATH, IMAGE_MODEL_PATH)

    mm = MatchMapper(audio_model, image_model)

    dir = './webapp/data'
    os.chdir(dir)

    mm.demo_with_animation('girl_lighthouse.jpg', 'utterance_380795.wav')
    print('finish')
