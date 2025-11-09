import torch
from torch.nn import Linear, BatchNorm1d  # Слои
from torch.nn import Sigmoid, Tanh, SiLU, ReLU  # Функции активации
from torch.nn import Module  # Модуль
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import time
import os
from oscillator_models import *


class Encoder(Module):
    def __init__(self, input_feature, output_feature, device):
        super(Encoder, self).__init__()
        self.device = device

        self.encoder_layer = Linear(in_features=input_feature, out_features=output_feature, bias=False, device=self.device)
        self.activation_function = SiLU()

    def forward(self, x):
        x = self.encoder_layer(x)
        x = self.activation_function(x)
        return x


class Decoder(Module):
    def __init__(self, input_feature, output_feature, device):
        super(Decoder, self).__init__()
        self.device = device

        self.output = output_feature
        self.decoder_layer = Linear(in_features=input_feature, out_features=output_feature, bias=False, device=self.device)
        self.activation_function = ReLU()

    def forward(self, x):
        x = self.decoder_layer(x)

        x = self.activation_function(x)
        return x


class AutoEncoder(Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Embedding(Module):
    def __init__(self, input_feature, output_feature, device="cpu"):
        super(Embedding, self).__init__()
        self.encoder = Encoder(input_feature=input_feature, output_feature=output_feature, device=device)

        self.__save_path_param = "parameters"
        self.__param_names = f"encoder_{input_feature}x{output_feature}.pth"

        if not os.path.isdir(self.__save_path_param):
            os.mkdir(self.__save_path_param)

        self.__exist_files = os.listdir(self.__save_path_param)

        if self.__param_names not in self.__exist_files:
            self.__train_and_save_params(output_feature, input_feature, device)
        else:
            self.encoder.load_state_dict(
                torch.load(self.__save_path_param + "/" + self.__param_names)
            )

    def forward(self, x):
        x = self.encoder(x)
        return x

    def __train_and_save_params(self, output_feature, input_feature, device):
        self.__decoder = Decoder(input_feature=output_feature, output_feature=input_feature, device=device)
        self.__autoencoder = AutoEncoder(encoder=self.encoder, decoder=self.__decoder)
        steps = 10000
        epochs = 100
        batch_size = 100

        _learning_loop(self.__autoencoder, batch_size, steps, epochs, device)
        # x = torch.tensor(data=[[10], [75], [5], [25], [0.0]], dtype=torch.float32, device=device)
        # y = self.__autoencoder(x)
        # print(y)
        del self.__decoder, self.__autoencoder

        torch.save(self.encoder.state_dict(), self.__save_path_param + "/" + self.__param_names)


class EnvironmentExperimental(Module):
    def __init__(self, oscillator, input_feature, output_feature, device="cpu"):
        super(EnvironmentExperimental, self).__init__()
        with torch.no_grad():
            self.input_layer = Embedding(
                input_feature=input_feature,
                output_feature=oscillator.volume(),
                device=device
            ).requires_grad_(requires_grad=False)

            self.reservoir = oscillator

        self.output_feature = Decoder(input_feature=oscillator.volume(), output_feature=output_feature, device=device)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.reservoir(x)
        x = self.output_feature(x)
        return x


def _learning_loop(autoencoder, batch_size, steps, epochs, device="cpu"):
    loss_fn = MSELoss()
    optim = Adam(params=autoencoder.parameters(), lr=1e-5)

    data = torch.randn(size=(steps, 1), device=device) * steps
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            y = autoencoder(x)

            loss = loss_fn(y, x)
            total_loss += loss.detach()

            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch: {epoch}, time: {time.time() - start_time:.2f}, loss: {total_loss / len(dataset) :.4f}")
            start_time = time.time()
    return autoencoder


def main():
    device = "cuda"

    emb = Embedding(1, 2**16, device=device)
    sub_1 = EnvironmentExperimental(
        oscillator=Sin_oscillator
    )


if __name__ == "__main__":
    main()