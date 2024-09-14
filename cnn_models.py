import torch
import torch.nn as nn


class CNN_casual(nn.Module):
    def __init__(self, random_state):
        super(CNN_casual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=32, kernel_size = (3,3)) # -4
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))# // 2
        
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3)) # -2
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))# // 2
        
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3)) # -2
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2))# // 2
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu5 = nn.ReLU()
        
        torch.manual_seed(random_state)
        self.dropout1 = nn.Dropout2d(p = .5)
        self.flatten1 = nn.Flatten(1)
        
        self.linear1 = nn.LazyLinear(out_features=256)
        self.relu2 = nn.ReLU()
        
        self.dropout2 = nn.Dropout2d(p = 0.2)
        self.linear2 = nn.Linear(in_features = 256, out_features = 2)
        self.softmax1 = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        
        x = self.dropout1(x)
        x = self.flatten1(x)
        
        x = self.linear1(x)
        x = self.relu2(x)
        
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.softmax1(x)
        return x

class CNN_casual_norm(nn.Module):
    def __init__(self, random_state):
        super(CNN_casual_norm, self).__init__()
        torch.manual_seed(random_state)
        self.cnn = nn.Sequential(
            
            nn.Conv2d(in_channels = 3, out_channels=32, kernel_size = (3,3)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3,3)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3)),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),

            nn.Dropout2d(p = .5),
            nn.Flatten(1),

            nn.LazyLinear(out_features=256),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),

            nn.Dropout2d(p = 0.5),
            nn.Linear(in_features = 256, out_features = 2),
            nn.LazyBatchNorm1d(),
            nn.Softmax(dim = 1),
        )
        
    def forward(self, x):
        return self.cnn(x)

class CNN_nin(nn.Module):
    def __init__(self):
        super(CNN_nin, self).__init__()
        self.cnn = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(), nn.Conv2d(32, 32, 1), nn.ReLU(), nn.Conv2d(32, 32, 1), nn.ReLU()),
            nn.MaxPool2d(kernel_size=2),
            
            self.nin_block(out_channels = 64, kernel_size=3, strides = 1, padding= 0),
            nn.MaxPool2d(kernel_size=2), 
            
            self.nin_block(out_channels = 128, kernel_size=3, strides = 1, padding= 0),
            nn.MaxPool2d(kernel_size=2),
        
            self.nin_block(out_channels = 256, kernel_size=3, strides = 1, padding= 0),
            nn.MaxPool2d(kernel_size=2),
        
            self.nin_block(out_channels = 256, kernel_size=3, strides = 1, padding= 0),
            nn.MaxPool2d(kernel_size=2),
        
            self.nin_block(out_channels = 2, kernel_size=3, strides = 1, padding =  0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Softmax(dim = 1)
            )

    def forward(self, x):
        return self.cnn(x)
        
    def nin_block(self, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())