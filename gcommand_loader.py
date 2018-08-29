import os
import os.path

import librosa
import numpy as np
import torch
import torch.utils.data as data

# Classes defined in the TensorFlow Speech Recognition Challenge.
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
           'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
           'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
           'up', 'visual', 'wow', 'yes', 'zero']

CLASSES1 = ['minnan', 'nanchang', 'kejia', 'changsha', 'shanghai', 'hebei', 'hefei', 'sichuan', 'shan3xi', 'ningxia']


AUDIO_EXTENSIONS = [
    '.wav', '.WAV', '.pcm',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_classes():
    classes = CLASSES1
    weight = None
    class_to_id = {classes[i]: i for i in range(len(classes))}
    return classes, weight, class_to_id

def get_segment(wav, seg_ini, seg_end):
    nwav = None
    if float(seg_end) > float(seg_ini):
        if wav[-1] == '|':
            nwav = wav + ' sox -t wav - -t wav - trim {} ={} |'.format(seg_ini, seg_end)
        else:
            nwav = 'sox {} -t wav - trim {} ={} |'.format(wav, seg_ini, seg_end)
    return nwav

def load_pcm(path, type, max=32768):
    pcm=np.fromfile(path, dtype=type)
    pcm_f=np.float32(pcm) / np.float32(max)
    return pcm_f

def make_dataset(dir, class_to_idx):
    spects = []
    print dir
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    #y, sr = librosa.load(path, sr=None)
    
    sr = 16000
    y = load_pcm(path, np.int16, max=32768)
    # n_fft = 512
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
   # print path, window, n_fft, sr, window_size, window_stride	
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        #classes, class_to_idx = find_classes(root)
        classes, _, class_to_idx = get_classes()
        spects = make_dataset(root, class_to_idx)
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return len(self.spects)

def param_loader(path, window_size, window_stride, window, normalize, max_len):
    y, sfr = librosa.load(path, sr=None)

    # window length
    win_length = int(sfr * window_size)
    hop_length = int(sfr * window_stride)
    n_fft = 512
    lowfreq = 20
    highfreq = sfr/2 - 400

    # melspectrogram
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
    D = np.abs(S)
    param = librosa.feature.melspectrogram(S=D, n_mels=40, fmin=lowfreq, fmax=highfreq, norm=None)

    # Add zero padding to make all param with the same dims
    if param.shape[1] < max_len:
        pad = np.zeros((param.shape[0], max_len - param.shape[1]))
        param = np.hstack((pad, param))

    # If exceeds max_len keep last samples
    elif param.shape[1] > max_len:
        param = param[:, -max_len:]

    param = torch.FloatTensor(param)

    # z-score normalization
    if normalize:
        mean = param.mean()
        std = param.std()
        if std != 0:
            param.add_(-mean)
            param.div_(std)

    return param

class Loader(data.Dataset):
    """A google commands data set loader using Kaldi data format::
    Args:
        root (string): Kaldi directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the param to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_id (dict): Dict with items (class_name, class_index).
        wavs (list): List of (wavs path, class_index) tuples
        STFT parameters: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=97):
        classes, weight, class_to_id = get_classes()
        wavs = make_dataset(root, class_to_id)
        if not wavs:
            raise RuntimeError("Found 0 segments in '" + root + "'. Folder should be in standard Kaldi format")  # pylint: disable=line-too-long

        self.root = root
        self.wavs = wavs
        self.classes = classes
        self.weight = torch.FloatTensor(weight) if weight is not None else None
        self.class_to_idx = class_to_id
        self.transform = transform
        self.target_transform = target_transform
        self.loader = param_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (params, target) where target is class_index of the target class.
        """
        key, path, target = self.wavs[index]
        params = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)  # pylint: disable=line-too-long
        if self.transform is not None:
            params = self.transform(params)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return key, params, target

    def __len__(self):
        return len(self.wavs)