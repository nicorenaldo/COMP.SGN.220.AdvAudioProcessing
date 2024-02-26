from pathlib import Path
import torch

from torch.nn import Module, Conv2d, MaxPool2d, \
    BatchNorm2d, ReLU, Linear, Sequential, Dropout, LSTM
from torchsummary import summary

from dataset import MyDataset


class MyModel(Module):
    def __init__(self,
                 output_classes: int,
                 ):
        super(MyModel, self).__init__()

        self.conv_block_1 = Sequential(
            Conv2d(in_channels=1,
                   out_channels=32,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            BatchNorm2d(num_features=32),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),
        )
        self.conv_block_2 = Sequential(
            Conv2d(in_channels=32,
                   out_channels=64,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            BatchNorm2d(num_features=64),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),
        )

        # LSTM layers for capturing time series data
        self.lstm = LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.dropout = Dropout(0.5)

        # Fully connected layer
        self.fc = Linear(128, output_classes)

    def forward(self, x):
        x = x if x.dim() == 4 else x.unsqueeze(1)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # Reshape for LSTM
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Use the output of the last LSTM time step
        x = self.fc(x)
        return x


if __name__ == '__main__':
    with open("dataset/features/labels.txt", "rb") as f:
        labels = f.read().decode("utf-8").split("\n")

    model = MyModel(output_classes=len(labels))
    train_ds = MyDataset(
        root_dir=Path("dataset/features"),
        split="train",
    )

    x, y = [], []
    for i in range(3):
        x_, y_ = train_ds[i]
        x.append(x_)
        y.append(y_)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    print(x.shape)

    d_time = 60
    d_feature = 1025
    batch_size = 3
    summary(
        model,
        input_size=(1, d_time, d_feature),
        batch_size=batch_size,
        device="cpu",
        branching=True
    )

    result = model(x)
    print("Result shape:", result.shape)
    print("Result:", result)
