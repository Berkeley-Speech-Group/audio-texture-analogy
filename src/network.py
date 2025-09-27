import torch.nn as nn
import torchvision
import torch
from nnAudio.Spectrogram import STFT
import torch.nn.functional as F

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()

        resnet = torchvision.models.resnet18(num_classes=256)
        # n_fft = 1024
        # window_size = 91
        # hop_size = 91  # stride
        n_fft = 512
        window_size = 400
        hop_size = 160  # stride
        self.AudioInconv = nn.Sequential(nn.Conv2d(1, 64, stride=2, kernel_size=7, padding=3, bias=False),
                                    resnet.bn1, resnet.relu, resnet.maxpool)
        self.AudioLayer1 = resnet.layer1
        self.AudioLayer2 = resnet.layer2
        self.AudioLayer3 = resnet.layer3
        self.AudioLayer4 = resnet.layer4
        self.AudioAvgpool = resnet.avgpool
        
        self.stft = STFT(n_fft=n_fft, win_length=window_size, hop_length=hop_size, sr=16000, output_format='Magnitude')

    def forward(self, audio):

        spec = self.stft(audio)
        a_f = self.AudioAvgpool(self.AudioLayer4(self.AudioLayer3(self.AudioLayer2(self.AudioLayer1(self.AudioInconv(spec.unsqueeze(1))))))).flatten(1)
        
        return a_f

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.f_a = nn.Linear(512, 512)
        self.f_i = nn.Linear(512, 512)

    def forward(self, audio, image):
        a_f = self.f_a(audio)
        i_f = self.f_i(image)

        f_f = torch.cat((a_f, i_f), dim=1)
        return f_f


class FusionPEModel(nn.Module):
    def __init__(self, dim):
        super(FusionPEModel, self).__init__()

        self.position_encoding = nn.Parameter(torch.zeros(1, 2, dim))
        self.f_1 = nn.Linear(512, 512)
        self.f_2 = nn.Linear(512, 512)


    def forward(self, cond1, cond2):

        f_1 = self.f_1(cond1)
        f_2 = self.f_2(cond2)
        cond = torch.stack([f_1, f_2], dim=1)
        cond = cond + self.position_encoding

        return cond




class FFNModel(nn.Module):
    def __init__(self):
        super(FFNModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512)  # Output size matches input size
        )

    def forward(self, audio, image):
        a_f = self.mlp(audio)
        i_f = self.mlp(image)

        f_f = torch.cat((a_f, i_f), dim=1)
        return f_f
