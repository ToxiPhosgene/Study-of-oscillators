from oscillator_models import *
import torch
from torch import concatenate
from matplotlib import pyplot as plt
import matplotlib
import random

matplotlib.use("QtAgg")


class NARMA(Oscillator):
    def __init__(self, time_series: int=10, period_of_variability: int=10, time_step: float=0.001, manual_seed=1):
        random.seed(manual_seed)
        super().__init__(time_step)
        self.period_of_variability = period_of_variability

        self.x = [0.0 for _ in range(time_series)]
        self.u = [random.random() / 2 for _ in range(time_series)]

        self.counter = 0

    def __update(self):
        x = 0.3 * self.x[-1] + 0.05 * self.x[-1] * sum(self.x[0:-2]) + self.u[-1] * self.u[0] + 0.1

        if self.counter >= self.period_of_variability / self.time_step:
            self.x.append(x)
            del self.x[0]

            self.u.append(random.random() / 2)
            del self.u[0]

            self.counter = 0

        self.counter += 1
        return x

    def __call__(self, *args, **kwargs):
        return self.__update()


def fourier_transform(sequence, period, device="cpu"):
    if type(sequence) != torch.Tensor:
        sequence = torch.tensor(data=sequence, device=device)

    magnitudes = []
    freq_bins = []
    for channel in sequence:
        fourier = torch.fft.rfft(channel)
        tmp_magnitudes = fourier.abs().unsqueeze(dim=0)
        tmp_freq_bins = torch.fft.rfftfreq(channel.size(0), period, device=device).unsqueeze(dim=0)
        tmp_magnitudes[:, 0] = 0
        tmp_freq_bins[:, 0] = 0

        magnitudes.append(tmp_magnitudes)
        freq_bins.append(tmp_freq_bins)

    magnitudes = concatenate(
        tensors=magnitudes,
        dim=0
    )
    magnitudes /= magnitudes.max()
    freq_bins = concatenate(
        tensors=freq_bins,
        dim=0
    )
    return [freq_bins, magnitudes]


def get_oscillogram(oscillator, delta: float, time_simulation: int, device: str="cpu"):
    step_simulation = int(time_simulation / oscillator.time_step)
    history = {"X": [], "mX": []}
    N = 60 / 0.001
    N = 2 / (N + 1)

    f_empty_history = True

    for _ in range(step_simulation):
        x = oscillator(delta)

        if f_empty_history:
            history["X"] = x
            history["mX"] = x
            f_empty_history = False
        else:
            # noinspection PyTypeChecker
            history["X"] = concatenate(tensors=(history["X"], x), dim=0)
            # noinspection PyTypeChecker
            history["mX"] = concatenate(
                tensors=(history["mX"], (x * N) + (history["mX"][-1] * (1 - N)))
            )

    keys = list(history.keys())
    for key in keys:
        history[key] = torch.transpose(history[key], dim0=1, dim1=0)

    return history


def static_analys(oscillator, delta, ax, simulation_time, device="cpu"):
    history = get_oscillogram(oscillator, delta, simulation_time, device=device)
    history["fX"] = fourier_transform(history["X"], oscillator.time_step, device=device)

    ax[0].plot(
        torch.transpose(history["X"], dim0=1, dim1=0).cpu().numpy(),
        linewidth=0.75
    )
    for freq_bins, magnitudes in zip(*history["fX"]):
        ax[1].plot(freq_bins.cpu().numpy(), magnitudes.cpu().numpy(), linewidth=0.75)


def _to_tensor(array):
    """
    Рекурсивно преобразует все из торча в нумпай
    :param array:
    :return:
    """
    # Если входное значения словарь
    if type(array) == dict:
        keys = list(array.keys())
        for key in keys:
            # если это торч, то переводим в нумпай, если нет, то рекурсивно вызываем
            if type(array[key]) == torch.Tensor:
                array[key] = array[key].cpu().numpy()
            else:
                array[key] = _to_tensor(array[key])

    # Если входное значения список
    elif type(array) == list:
        for i in range(len(array)):
            # если это торч, то переводим в нумпай, если нет, то рекурсивно вызываем
            if type(array[i]) == torch.Tensor:
                array[i] = array[i].cpu().numpy()
            else:
                array[i] = _to_tensor(array[i])

    return array


def dynamic_analys(oscillator, delta: list, simulation_time: list):
    if len(delta) != len(simulation_time):
        raise ValueError("Число управляющих значений не равно числу временных промежутков")

    history = {"X": [], "mX": []}
    for _delta, _time in zip(delta, simulation_time):
        tmp_history = get_oscillogram(oscillator, _delta, _time)
        history["X"].append(tmp_history["X"])
        history["mX"].append(tmp_history["mX"])

    # noinspection PyTypeChecker
    history["X"] = concatenate(
        tensors=history["X"],
        dim=1
    )
    # noinspection PyTypeChecker
    history["mX"] = concatenate(
        tensors=history["mX"],
        dim=1
    )

    history["fX"] = fourier_transform(history["X"], period=oscillator.time_step)

    history["meanX"] = []
    sum = 0
    for channel in history["X"]:
        f_empty_dict = True
        for i, item in enumerate(channel):
            sum += item
            if f_empty_dict:
                # noinspection PyTypeChecker
                tmp_history = torch.unsqueeze(sum / (i + 1), dim=0)
                f_empty_dict = False
            else:
                # noinspection PyTypeChecker
                tmp_history = concatenate(
                    tensors=[tmp_history, torch.unsqueeze(sum / (i + 1), dim=0)],
                    dim=-1
                )
        history["meanX"].append(tmp_history.unsqueeze(dim=0))

    history["meanX"] = concatenate(tensors=history["meanX"], dim=0)
    history = _to_tensor(history)
    return history


def example_research():

    oscillator_used = Chua_oscillator()

    fig, ax = plt.subplots(2, 1)

    simulation_time = 60
    static_analys(oscillator_used, 0.0, ax, simulation_time)
    static_analys(oscillator_used, 1.0, ax, simulation_time)

    ax[0].grid()
    ax[1].grid()

    ax[0].set_xlim(0)
    ax[1].set_xlim(0, 2)

    ax[0].set_title("а) Осциллограмма", loc="left")
    ax[0].set_xlabel("Время, 1 мс", fontsize=10, loc="right")
    ax[0].set_ylabel("Амплитуда")

    ax[1].set_title("б) Частотный спектр", loc="left")
    ax[1].set_xlabel("Частота, Гц", fontsize=10, loc="right")
    ax[1].set_ylabel("Амплитуда")

    ax[0].legend(["delta = 0.0", "delta = 1.0"])
    ax[1].legend(["delta = 0.0", "delta = 1.0"])

    plt.show()

    fig, ax = plt.subplots(2, 2)

    key_used = "meanX"

    graphs = {"а) Сравнительный график": [],
              "б) График высокочастотных импульсов (1 с импульса, 1 с релаксации)":
                  dynamic_analys(oscillator_used, [0.0, *[0.5 if i % 2 == 0 else 0 for i in range(0, 180)], 0.0], ax)[key_used],

              "в) График длительного импульса (180 с)":
                  dynamic_analys(oscillator_used, [0.0, 0.5, 0.0], ax)[key_used],

              "г) График низкочастотных импульсов (10 с импульс, 10 с релаксации)":
                  dynamic_analys(oscillator_used, [0.0, *[0.5 if i % 2 == 0 else 0 for i in range(0, 18)], 0.0], ax)[key_used]
              }

    keys = list(graphs.keys())
    for key in keys[1:]:
        graphs[keys[0]].append(graphs[key])

    for item in graphs[keys[0]]:
        ax[0, 0].plot(item, linewidth=0.75)
    ax[0, 0].legend(keys[1:])
    ax[0, 1].plot(graphs[keys[1]], linewidth=0.75)
    ax[1, 0].plot(graphs[keys[2]], linewidth=0.75)
    ax[1, 1].plot(graphs[keys[3]], linewidth=0.75)

    ax[0, 0].set_title(keys[0], loc="left")
    ax[0, 1].set_title(keys[1], loc="left")
    ax[1, 0].set_title(keys[2], loc="left")
    ax[1, 1].set_title(keys[3], loc="left")

    ax[0, 0].set_xlabel("Время, 1 мс", loc="right")
    ax[0, 0].set_ylabel("Амплитуда")
    ax[0, 1].set_xlabel("Время, 1 мс", loc="right")
    ax[0, 1].set_ylabel("Амплитуда")
    ax[1, 0].set_xlabel("Время, 1 мс", loc="right")
    ax[1, 0].set_ylabel("Амплитуда")
    ax[1, 1].set_xlabel("Время, 1 мс", loc="right")
    ax[1, 1].set_ylabel("Амплитуда")

    ax[0, 0].grid()
    ax[0, 0].set_xlim(0)
    ax[0, 1].grid()
    ax[0, 1].set_xlim(0)
    ax[1, 0].grid()
    ax[1, 0].set_xlim(0)
    ax[1, 1].grid()
    ax[1, 1].set_xlim(0)

    plt.show()


def main():
    history = {"NARMA5": [], "NARMA10":[]}
    narma = NARMA(time_series=5, period_of_variability=1)

    for i in range(int(60 // narma.time_step)):
        history["NARMA5"].append(narma())

    narma = NARMA(time_series=10, period_of_variability=1)

    for i in range(int(60 // narma.time_step)):
        history["NARMA10"].append(narma())

    plt.plot(history["NARMA5"])
    plt.plot(history["NARMA10"])
    plt.show()


if __name__ == "__main__":
    example_research()
    # main()