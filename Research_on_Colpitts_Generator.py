from tools_for_study import static_analys, dynamic_analys
from oscillator_models import *
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("QtAgg")
oscillator = Colpitts_generator


def block_static_analys():
    fig, ax = plt.subplots(2, 1)
    simulation_time = 180

    # Статическая симуляция
    seq_delta = [0.0, 0.25, 0.5, 0.75, 1.0]
    for delta in seq_delta:
        static_analys(oscillator=oscillator, delta=delta, ax=ax, simulation_time=simulation_time)

    # Отображение статического наблюдения
    ax[0].grid()
    ax[1].grid()

    ax[0].set_xlim(0)
    ax[1].set_xlim(0, 0.15)

    ax[0].set_title("а) Осциллограмма", loc="left")
    ax[0].set_xlabel("Время, 1 мс", fontsize=10, loc="right")
    ax[0].set_ylabel("Амплитуда")

    ax[1].set_title("б) Частотный спектр", loc="left")
    ax[1].set_xlabel("Частота, Гц", fontsize=10, loc="right")
    ax[1].set_ylabel("Амплитуда")

    ax[0].legend([f"delta = {item}" for item in seq_delta])
    ax[1].legend([f"delta = {item}" for item in seq_delta])

    plt.show()


def main():
    # block_static_analys()

    fig, ax = plt.subplots(2, 2)

    # Протокол выполнения эксперимента
    experiment = [
        {
            "name":     "Короткий, длинный, пауза, короткий",
            "delta":    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "sim_time": [180, 12., 60., 120, 60., 60., 60., 12., 60.]
        },
        {
            "name":     "Длинный, короткий, пауза, короткий",
            "delta":    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "sim_time": [180, 120., 60., 12, 60., 60., 60., 12., 60.]
        },
        {
            "name":     "Короткий, пауза, короткий, длинный",
            "delta":    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "sim_time": [180, 12., 60., 60., 60., 12., 60., 120, 60.]
        }
    ]

    graphs_for_study = {
        parameters["name"]: dynamic_analys(oscillator=oscillator, delta=parameters["delta"],
                                           simulation_time=parameters["sim_time"]) for parameters in experiment
    }

    for i, parameters in enumerate(experiment):
        _sim_seq = []
        for sim_time, delta in zip(parameters["sim_time"], parameters["delta"]):
            _tmp_time = [delta for _ in range(0, int(sim_time // 0.001))]
            _sim_seq.extend(_tmp_time)
        experiment[i]["__time_for_graph"] = _sim_seq

    instrument = "mX"

    ax[0, 0].plot(graphs_for_study[experiment[0]["name"]][instrument])
    ax[0, 0].plot(experiment[0]["__time_for_graph"])
    ax[0, 0].legend(["График средней осциллятора", "Стимульный сигнал"])
    ax[0, 0].grid()
    ax[0, 0].set_title(f"а) {experiment[0]["name"]}")

    ax[0, 1].plot(graphs_for_study[experiment[1]["name"]][instrument])
    ax[0, 1].plot(experiment[1]["__time_for_graph"])
    ax[0, 1].legend(["График средней осциллятора", "Стимульный сигнал"])
    ax[0, 1].grid()
    ax[0, 1].set_title(f"б) {experiment[1]["name"]}")

    ax[1, 0].plot(graphs_for_study[experiment[2]["name"]][instrument])
    ax[1, 0].plot(experiment[2]["__time_for_graph"])
    ax[1, 0].legend(["График средней осциллятора", "Стимульный сигнал"])
    ax[1, 0].grid()
    ax[1, 0].set_title(f"в) {experiment[2]["name"]}")

    ax[1, 1].plot(graphs_for_study[experiment[0]["name"]][instrument])
    ax[1, 1].plot(graphs_for_study[experiment[1]["name"]][instrument])
    ax[1, 1].plot(graphs_for_study[experiment[2]["name"]][instrument])
    ax[1, 1].legend(["а", "б", "в"])
    ax[1, 1].grid()
    ax[1, 1].set_title("г) Сравнительный график")

    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(graphs_for_study[experiment[0]["name"]][instrument])
    ax.plot(graphs_for_study[experiment[1]["name"]][instrument])
    ax.plot(graphs_for_study[experiment[2]["name"]][instrument])
    ax.legend(["а", "б", "в"])
    ax.grid()
    ax.set_title("Сравнительный график")
    plt.show()


if __name__ == "__main__":
    main()