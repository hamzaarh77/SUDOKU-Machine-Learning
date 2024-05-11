import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# class Conv2dSame(th.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=th.nn.ReflectionPad2d):
#         super(Conv2dSame, self).__init__()
#         ka = kernel_size // 2
#         kb = ka - 1 if kernel_size % 2 == 0 else ka
#         self.net = nn.Sequential(
#             padding_layer((ka,kb,ka,kb)),
#             nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )
#     def forward(self, x):
#         return self.net(x)


# class SudokuCNN(nn.Module):
#     def __init__(self):
#         super(SudokuCNN, self).__init__()
#         self.conv_layers = nn.Sequential(
#             Conv2dSame(9, 64, 3),    # Couche 1
#             Conv2dSame(64, 64, 3),  # Couche 2
#             Conv2dSame(64, 64, 3),  # Couche 2

#             #Conv2dSame(64, 128, 3), # Couche 3

#             #Conv2dSame(128, 128, 3), # Couche 4
        
#         )
#         self.last_conv = nn.Conv2d(64, 9, 1)  # Chaque sortie de grille prédit un chiffre pour chaque position (1-9)
    
#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.last_conv(x)
#         test = F.log_softmax(x, dim=1) 
#         return test # softmax pour normaliser les prédictions



class Conv2dSameResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d, dropout_rate=0.5):
        super(Conv2dSameResidual, self).__init__()
        self.net = nn.Sequential(
            padding_layer((kernel_size//2, kernel_size//2 - (kernel_size % 2 == 0), kernel_size//2, kernel_size//2 - (kernel_size % 2 == 0))),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )
        self.residual = (in_channels == out_channels)

    def forward(self, x):
        if self.residual:
            return x + self.net(x)
        else:
            return self.net(x)

class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            Conv2dSameResidual(9, 32, 3),    # Couche 1
            Conv2dSameResidual(32, 32, 3),   # Couche 2
            Conv2dSameResidual(32, 64, 3),   # Couche 3
        )
        self.last_conv = nn.Conv2d(64, 9, 1)  # Chaque sortie de grille prédit un chiffre pour chaque position (1-9)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.last_conv(x)
        return F.log_softmax(x, dim=1)  # softmax pour normaliser les prédictions


        