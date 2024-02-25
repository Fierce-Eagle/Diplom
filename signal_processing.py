from matplotlib import pyplot as plt


def load_info_from_file(filename, is_show_info=False):
    """
    Загрузка сигналов и частоты
    @param filename:
    @param is_show_info:
    @return:
    """
    channel_1 = []
    channel_2 = []
    channel_3 = []

    with open(filename) as file:
        for _ in range(7):  # скип ненужных строк
            line = file.readline()
            print(line)

        line = file.readline()
        params = line.split()
        print(params)
        freqence = float(params[3])

        for _ in range(12):  # скип ненужных строк
            try:
                file.readline()
            except:
                pass

        for line in file:  # определение минимальной дисперсии
            channel_params = line.split()
            channel_1.append(float(channel_params[1]))
            channel_2.append(float(channel_params[2]))
            channel_3.append(float(channel_params[3]))

    if is_show_info:
        print("Частота:", freqence)
        plt.plot(channel_1, label="сигнал 1")
        plt.plot(channel_2, label="сигнал 2")
        plt.plot(channel_3, label="сигнал 3")
        plt.legend()
        plt.show()

    return channel_1, channel_2, channel_3, freqence


def cut_empty_frames_in_signals(signal_1, signal_2, signal_3):
    pass
