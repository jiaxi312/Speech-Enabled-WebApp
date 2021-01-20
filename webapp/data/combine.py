import ffmpeg

import moviepy.editor as mpe

my_clip = mpe.VideoFileClip('./video2.mp4')
audio_background = mpe.AudioFileClip('./utterance_380795.wav')
final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
final_clip = my_clip.set_audio(final_audio)

