import numpy as np
import pandas as pd
import scipy
import my_dataset
from matplotlib import pyplot as plt

def split_data_to_dataframe(channel_1, channel_2, channel_3, label, frame_size=5000, frame_stride=2000, smoothing_window=10, spectrum_length=500):
    """
    dataframe: сигнал_1, сигнал_2, сигнал_3, сигнал_1 - сигнал_2, сигнал_1 - сигнал_3, сигнал_2 - сигнал_3
    smoothing_window: уменьшается спектр на значение размера окна - 1
    """
    new_dataset = pd.DataFrame()
    length = len(channel_1) - frame_size - frame_stride
    i = 0
    while i < length:
        temp_ch_1 = channel_1[i:i+frame_size]
        temp_ch_2 = channel_2[i:i+frame_size]
        temp_ch_3 = channel_3[i:i+frame_size]

        # построение спектров по каналам
        spectrum_1, spectrum_2, spectrum_3, diff_1, diff_2, diff_3 = create_smoothed_spectrum(temp_ch_1, temp_ch_2, temp_ch_3, smoothing_window, spectrum_length)

        # построение фрейма
        frame = np.concatenate([spectrum_1, spectrum_2, spectrum_3, diff_1, diff_2, diff_3])
        new_dataset = pd.concat([new_dataset, pd.DataFrame({"data": [frame], "label": [label]})], ignore_index=True)

        i += frame_stride

    return new_dataset

def create_smoothed_spectrum(ch_1, ch_2, ch_3, smoothing_window, spectrum_length):
    magnitude_length = spectrum_length + smoothing_window - 1

    #plt.figure(figsize=(10, 6))

    fft_res_1 = np.fft.fft(ch_1)
    fft_res_2 = np.fft.fft(ch_2)
    fft_res_3 = np.fft.fft(ch_3)
    #print(len(fft_res_1))

    magnitude_spectrum_1 = np.abs(fft_res_1)[:magnitude_length]
    magnitude_spectrum_2 = np.abs(fft_res_2)[:magnitude_length]
    magnitude_spectrum_3 = np.abs(fft_res_3)[:magnitude_length]
    # print(len(magnitude_spectrum_1))

    smoothed_spectrum_1 = np.convolve(magnitude_spectrum_1, np.ones(smoothing_window) / smoothing_window,
                                      mode='valid')  # сглаживание
    smoothed_spectrum_2 = np.convolve(magnitude_spectrum_2, np.ones(smoothing_window) / smoothing_window,
                                      mode='valid')  # сглаживание
    smoothed_spectrum_3 = np.convolve(magnitude_spectrum_3, np.ones(smoothing_window) / smoothing_window,
                                      mode='valid')  # сглаживание
    # print()
    # print(len(smoothed_spectrum_1))
    # print(len(smoothed_spectrum_2))
    # print(len(smoothed_spectrum_3))
    diff_1 = smoothed_spectrum_1 - smoothed_spectrum_2
    diff_2 = smoothed_spectrum_1 - smoothed_spectrum_3
    diff_3 = smoothed_spectrum_2 - smoothed_spectrum_3
    # print(smoothed_spectrum_1[:10])
    # print(smoothed_spectrum_2[:10])
    # print(diff_1[:10])
    # print(len(diff_1))
    # print(len(diff_2))
    # print(len(diff_3))

    #plt.plot(magnitude_spectrum_1)
    #plt.plot(magnitude_spectrum_2)
    #plt.plot(magnitude_spectrum_3)
    # plt.title('Сглаженные спектры сигналов из нескольких колонок (скользящее среднее)')
    # plt.xlabel('Частота (Гц)')
    # plt.ylabel('Амплитуда')
    # plt.legend()
    # plt.grid(True)
    #plt.show()

    return smoothed_spectrum_1, smoothed_spectrum_2, smoothed_spectrum_3, diff_1, diff_2, diff_3

def plot_smoothed_spectrum_multiple_columns(data_frame, sampling_rate=5000, window_size=10, column_numbers=8, cutoff_frequency=500):
    """
    Построение сглаженных спектров данных для нескольких колонок.

    Parameters:

    data_frame (pd.DataFrame): DataFrame с данными.
    sampling_rate (float): Частота дискретизации данных.
    window_size (int): Размер окна скользящего среднего.
    column_numbers (list): Список номеров колонок для построения сглаженных спектров.

        Returns:

    None"""


    # plt.figure(figsize=(10, 6))

    spectrums = []

    for column_number in column_numbers:  # перебор по колонкам
        column_name = data_frame.columns[column_number]  # взяли 1 канал
        data = data_frame[column_name].values  # достали значения

        n = len(data)  # взяли длинну 1 -> 3 -> 5к?
        fft_result = np.fft.fft(data)  # построили спектр по сигналу
        frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)  # взяли частоту 3 000 / 5 000 = 0.6?

        # Игнорируем отрицательные частоты
        positive_frequencies = frequencies[:n // 2]  # n // 2 = 2500k? можно вынести в const, сначала идут разве положительные частоты?

        # Применение скользящего среднего
        magnitude_spectrum = np.abs(fft_result)[:n // 2]  # аналогично выше
        smoothed_spectrum = np.convolve(magnitude_spectrum, np.ones(window_size) / window_size, mode='valid')  # сглаживание

        # Определение индекса частоты, соответствующей cutoff_frequency
        cutoff_index = np.where(positive_frequencies >= cutoff_frequency)[0][0]  # частота 500, индекс как правило с 10

        # Отрезкаем значения после cutoff_frequency
        positive_frequencies = positive_frequencies[:cutoff_index]
        smoothed_spectrum = smoothed_spectrum[:cutoff_index]

        # Построение сглаженного спектра
        # plt.plot(positive_frequencies[:len(smoothed_spectrum)], smoothed_spectrum, label=f'Колонка {column_name}')

        spectrums.append(smoothed_spectrum)

    # plt.title('Сглаженные спектры сигналов из нескольких колонок (скользящее среднее)')
    # plt.xlabel('Частота (Гц)')
    # plt.ylabel('Амплитуда')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return spectrums