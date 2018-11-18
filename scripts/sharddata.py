import wave
import contextlib
import numpy as np

def get_length(wav_path):
    with contextlib.closing(wave.open(wav_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

class Example(object):
    def __init__(self, duration, line):
        self.duration = duration
        self.line = line

examples = []
train_files = ["librivox-train-clean-100.csv", "librivox-train-clean-360.csv", "librivox-train-other-500.csv"]
for train_file in train_files:
    with open(train_file) as f:
        lines = [line.strip() for line in f]
    
    first_line = True
    for line in lines:
        if first_line:
            first_line = False
            continue
        wav_path = line.split(',')[0]
        wav_len = get_length(wav_path)
        if wav_len <= 16.7:
            examples.append( Example(wav_len, line) )

examples = sorted(examples, key=lambda x: x.duration)

remainder = len(examples) % 8
examples = examples[:-remainder]

blocksize = len(examples) / 8

index = 0
for example in examples:
    suffix = index // blocksize
    fname = "part%d.csv" % suffix
    with open(fname, 'a+') as f:
        f.write(example.line + "\n")
    index +=1

