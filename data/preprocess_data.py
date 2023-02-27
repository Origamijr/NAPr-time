import argparse
import os, glob
import librosa

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='../datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed/', type=str)
parser.add_argument('--pattern', default='*_mic1.flac', type=str)
args = parser.parse_args()

for dir in os.listdir(args.input):
    if os.path.isdir(os.path.join(args.input, dir)):
        for file in glob.glob(os.path.join(args.input, dir) + '/' + args.pattern):
            y, _ = librosa.load(file, sr=16000)
            print(y.shape)
            break