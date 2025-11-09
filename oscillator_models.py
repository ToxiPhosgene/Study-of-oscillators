import math

import torch
from sympy.physics.units import volume
from torch import tensor, rand


class Oscillator:
    """
    Перенеси его к остальным осцилляторам. Это эталон. И переделай остальные осцилляторы под торч
    """
    def __init__(self, volume=1, time_step=0.001, generator=None, seed=111, device="cpu"):
        self.time_step = time_step
        self._volume = volume
        self.device = device

        if generator == None:
            self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def __call__(self, x):
        return x

    def size(self):
        return None


class Chua_oscillator(Oscillator):
    """
    Осциллятора Чжуа.
    """
    def __init__(self, x=0.1, y=0.0, z=0.0, alpha=16.8, beta=28.0, c=-0.35, **kwargs):
        super().__init__(**kwargs)

        # Основные параметры
        self.x = tensor(
            data=[x for _ in range(self._volume)],
            device=self.device
        ) + (rand(
            size=(1, self._volume),
            device=self.device,
            generator=self.generator
        ) * 2 - 1) * x

        self.y = tensor(
            data=[y for _ in range(self._volume)],
            device=self.device
        ) + (rand(
            size=(1, self._volume),
            device=self.device,
            generator=self.generator
        ) * 2 - 1) * y

        self.z = tensor(
            data=[z for _ in range(self._volume)],
            device=self.device
        ) + (rand(
            size=(1, self._volume),
            device=self.device,
            generator=self.generator
        ) * 2 - 1) * z

        # Параметры, ответственные за изменение основных параметров
        self.alpha = tensor(
            data=[alpha for _ in range(self._volume)], device=self.device
        ) + (rand(
            size=(1, self._volume),
            device=self.device,
            generator=self.generator
        ) * 2 - 1) * alpha * 1e-3

        self.beta = tensor(
            data=[beta for _ in range(self._volume)], device=self.device
        ) + (rand(
            size=(1, self._volume),
            device=self.device,
            generator=self.generator
        ) * 2 - 1) * beta * 1e-3

        self.c = tensor(
            data=[c for _ in range(self._volume)], device=self.device
        ) + (rand(
            size=(1, self._volume),
            device=self.device,
            generator=self.generator
        ) * 2 - 1) * c * 1e-3

    def __update(self, delta=0.0):
        # Расчет изменения состояния осциллятора
        __dx = (self.alpha * (self.y - self.x ** 3 - self.c * self.x) + delta) * self.time_step
        __dy = (self.x - self.y + self.z) * self.time_step
        __dz = (-self.beta * self.y) * self.time_step

        # Изменение основных параметров
        self.x += __dx
        self.y += __dy
        self.z += __dz

        return self.x

    def __call__(self, delta, *args, **kwargs):
        x = self.__update(delta)
        return x


class Colpitts_generator(Oscillator):
    """
    Генератор Колпитца.
    ToDo:
    """
    def __init__(self, a=60, b=0.78, c=7.68, d=0.08, e=7.5):
        super().__init__()

        # Параметры, ответственные за изменение основных параметров
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

        # Основные параметры осциллятора
        self.x = 0.8
        self.y = 0.3
        self.z = 6.7

    def __F(self, z):
        """
        Особая функция генератора Колпитца. Ответственна за нелинейность, но из-за нее может уменьшиться производительность.
        :param z:
        :return:
        """
        if z <= (self.e - 1):
            return self.e - z - 1
        else:
            return 0

    def __update(self, delta=0.0):
        dx = (self.y - self.a * self.__F(self.z) + delta) * self.time_step
        dy = (self.c - self.x - self.b * self.y - self.z) * self.time_step
        dz = (self.y - self.d * self.z) * self.time_step

        self.x += dx
        self.y += dy
        self.z += dz

        return self.x

    def __call__(self, delta, *args, **kwargs):
        x = self.__update(delta)
        return x


class Sin_oscillator(Oscillator):
    def __init__(self, a = 1):
        super().__init__()
        self.x = 0
        self.y = 0
        self.a = 200 * a / 3.14

    def __update(self):
        dx = math.sin(self.y)
        dy = self.a * self.time_step

        self.x += dx
        self.y += dy
        return self.x

    def __call__(self, *args, **kwargs):
        return self.__update()


if __name__ == "__main__":
    osc = Chua_oscillator(volume=8)
    x = tensor(data=[0.0 for _ in range(8)])
    ten = tensor(data=[0.0, 0.0], dtype=torch.float32, device="cuda")