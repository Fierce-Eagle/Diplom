import numpy as np


def load_info_from_file(filename):
    """
    Загрузка сигналов и частоты
    @param filename:
    @return:
    """

    channel_1 = []
    channel_2 = []
    channel_3 = []

    with open(filename) as file:
        for _ in range(7):  # скип ненужных строк
            file.readline()

        line = file.readline()
        params = line.split()
        frequency = float(params[4]) * 1000

        for _ in range(12):  # скип ненужных строк
            file.readline()

        for line in file:  # определение минимальной дисперсии
            channel_params = line.split()
            channel_1.append(float(channel_params[1]))
            channel_2.append(float(channel_params[2]))
            channel_3.append(float(channel_params[3]))

    return channel_1, channel_2, channel_3, frequency


def cut_empty_frames_in_signals(signal_1, signal_2, signal_3, frequency):
    noise_step = int(0.1 * frequency)
    noise_signal_1 = max(np.abs(signal_1[100:noise_step]))
    noise_signal_2 = max(np.abs(signal_2[100:noise_step]))
    noise_signal_3 = max(np.abs(signal_3[100:noise_step]))

    new_signal_1 = []
    new_signal_2 = []
    new_signal_3 = []

    for sg_1, sg_2, sg_3 in zip(signal_1, signal_2, signal_3):
        if abs(sg_1) > noise_signal_1 and abs(sg_2) > noise_signal_2 and abs(sg_3) > noise_signal_3:
            new_signal_1.append(sg_1)
            new_signal_2.append(sg_2)
            new_signal_3.append(sg_3)

    return new_signal_1, new_signal_2, new_signal_3
