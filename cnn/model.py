import math
import torch
import torch.nn as nn


class CNNTextClassifier(nn.ModuleList):

    def __init__(self, params):
        super().__init__()

        self.seq_len = params.seq_len
        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.kernel_lengths = list(params.kernel_lengths)
        self.convolvers = []
        self.poolers = []
        self.strides = getattr(params, 'strides')
        if not self.strides:
            self.strides = [params.stride] * len(self.kernel_lengths)

        self.dropout = nn.Dropout(0.25)

        self.conv_output_size = params.conv_output_size
        self.stride = params.stride

        # Embedding layer definition
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0)

        # default: 4 CNN layers with max pooling
        for i, (kernel_len, stride) in enumerate(zip(self.kernel_lengths, self.strides)):
            self.convolvers.append(nn.Conv1d(self.seq_len, self.conv_output_size, kernel_len, stride))
            # setattr(self, f'conv_{i + 1}', self.convolvers[i])
            self.poolers.append(nn.MaxPool1d(kernel_len, stride))
            # setattr(self, f'pool_{i + 1}', self.poolers[i])

        self.encoding_size = self.cnn_output_size()
        print(f'encoding_size: {self.encoding_size}')
        # Fully connected layer definition
        self.linear_layer = nn.Linear(self.encoding_size, 1)

    def cnn_output_size(self):
        """ Calculate the number of encoding dimensions output from CNN layers

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        """
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_lengths[0] - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_lengths[0] - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        out_pool_total = 0
        for kernel_len, stride in zip(self.kernel_lengths, self.strides):
            out_conv = ((self.embedding_size - 1 * (kernel_len - 1) - 1) / stride) + 1
            out_conv = math.floor(out_conv)
            out_pool = ((out_conv - 1 * (kernel_len - 1) - 1) / stride) + 1
            out_pool = math.floor(out_pool)
            out_pool_total += out_pool

        # Returns "flattened" vector (input for fully connected layer)
        return out_pool_total * self.conv_output_size

    def forward(self, x):
        """ Takes sequence of integers (token indices) and outputs binary class label """

        x = self.embedding(x)

        conv_outputs = []
        for (conv, pool) in zip(self.convolvers, self.poolers):
            z = conv(x)
            z = torch.relu(z)
            z = pool(z)
            conv_outputs.append(z)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat(conv_outputs, 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.linear_layer(union)
        # Dropout is applied
        out = self.dropout(out)
        # Activation function is applied
        out = torch.sigmoid(out)

        return out.squeeze()
