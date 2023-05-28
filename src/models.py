import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN_v1(nn.Module):
    def __init__(
        self,
        num_classes,
        cnn_output_height=4,
        gru_hidden_size=128,
        gru_num_layers=2,
        cnn_output_width=32,
    ):
        super(CRNN_v1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
        self.norm4 = nn.InstanceNorm2d(64)
        self.gru_input_size = cnn_output_height * 64
        self.gru = nn.GRU(
            self.gru_input_size,
            gru_hidden_size,
            gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

        self.cnn_output_width = cnn_output_width

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.leaky_relu(out)
        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.gru_input_size)
        out, _ = self.gru(out)  # torch.Size([32, 32, 256])
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])]
        )
        return out


class CRNN_v2(nn.Module):
    def __init__(
        self,
        num_classes,
        cnn_output_height=4,
        rnn_hidden_size=128,
        rnn_num_layers=2,
        cnn_output_width=14,
    ):
        super(CRNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=0)

        self.rnn_input_size = cnn_output_height * 64
        self.blstm1 = nn.LSTM(
            self.rnn_input_size,
            rnn_hidden_size,
            rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.blstm2 = nn.LSTM(
            self.rnn_input_size,
            rnn_hidden_size,
            rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

        self.cnn_output_width = cnn_output_width

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.maxpool2(out)
        out = F.relu(out)
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.maxpool3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.conv5(out)
        out = self.norm5(out)
        out = self.maxpool5(out)
        out = self.conv6(out)

        out = out.permute(0, 3, 2, 1)
        out = out.reshape(batch_size, -1, self.rnn_input_size)
        out, _ = self.blstm1(out)
        out, _ = self.blstm2(out)  # (N=32, L=14, D∗H=256)
        out = torch.stack(
            [F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])]
        )
        return out
