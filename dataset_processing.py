import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from scipy import fft as fft
from matplotlib import pyplot as plt
from global_params import *


def split_data_to_dataframe(channel_1, channel_2, channel_3, label, frame_size=5000, frame_stride=2000, frame_coeff=1,
                            smoothing_window=10, spectrum_length=500):
    """
    создание спектров из сигналов
    dataframe: сигнал_1, сигнал_2, сигнал_3, сигнал_1 - сигнал_2, сигнал_1 - сигнал_3, сигнал_2 - сигнал_3
    smoothing_window: уменьшается спектр на значение размера окна - 1
    """
    new_dataset = pd.DataFrame()
    signals_1 = []
    signals_2 = []
    signals_3 = []
    spectr_1 = []
    spectr_2 = []
    spectr_3 = []
    spectrums = []
    length = len(channel_1) - frame_size - frame_stride
    i = 0
    while i < length:
        temp_ch_1 = channel_1[i:i + frame_size]
        temp_ch_2 = channel_2[i:i + frame_size]
        temp_ch_3 = channel_3[i:i + frame_size]

        # построение спектров по каналам
        spectrum_1, spectrum_2, spectrum_3, diff_1, diff_2, diff_3 = create_smoothed_spectrum(temp_ch_1, temp_ch_2,
                                                                                              temp_ch_3,
                                                                                              smoothing_window,
                                                                                              spectrum_length,
                                                                                              frame_coeff)
        # построение фрейма
        frame = np.concatenate([spectrum_1, spectrum_2, spectrum_3, diff_1, diff_2, diff_3])

        signals_1.append(temp_ch_1)
        signals_2.append(temp_ch_2)
        signals_3.append(temp_ch_3)
        spectr_1.append(spectrum_1)
        spectr_2.append(spectrum_2)
        spectr_3.append(spectrum_3)
        spectrums.append(frame)

        if GlobalParams.show_spectrum == True:
            GlobalParams.show_spectrum = False
            print("спектр:")
            plt.figure(figsize=(15, 6))
            plt.plot(frame)
            plt.show()

        new_dataset = pd.concat([new_dataset, pd.DataFrame({"data": [frame], "label": [label]})], ignore_index=True)

        i = i + frame_size - frame_stride

    return new_dataset, signals_1, signals_2, signals_3, spectr_1, spectr_2, spectr_3, spectrums


def create_smoothed_spectrum(ch_1, ch_2, ch_3, smoothing_window, spectrum_length, frame_coeff):
    magnitude_length = spectrum_length + smoothing_window - 1

    # взять часть сигнала
    signal_path_size = len(ch_1) // frame_coeff

    lst_1 = []
    lst_2 = []
    lst_3 = []
    for i in range(frame_coeff):
        lst_1.append(ch_1[(i * signal_path_size):(i + 1) * signal_path_size])
        lst_2.append(ch_2[(i * signal_path_size):(i + 1) * signal_path_size])
        lst_3.append(ch_3[(i * signal_path_size):(i + 1) * signal_path_size])

    mean_ch_1 = np.mean(lst_1, axis=0)
    mean_ch_2 = np.mean(lst_2, axis=0)
    mean_ch_3 = np.mean(lst_3, axis=0)

    fft_res_1 = fft.fft(mean_ch_1)
    fft_res_2 = fft.fft(mean_ch_2)
    fft_res_3 = fft.fft(mean_ch_3)

    magnitude_spectrum_1 = np.abs(fft_res_1)[:magnitude_length]
    magnitude_spectrum_2 = np.abs(fft_res_2)[:magnitude_length]
    magnitude_spectrum_3 = np.abs(fft_res_3)[:magnitude_length]

    # сглаживать после объединения
    smoothed_spectrum_1 = np.convolve(magnitude_spectrum_1, np.ones(smoothing_window) / smoothing_window,
                                      mode='valid')  # сглаживание
    smoothed_spectrum_2 = np.convolve(magnitude_spectrum_2, np.ones(smoothing_window) / smoothing_window,
                                      mode='valid')  # сглаживание
    smoothed_spectrum_3 = np.convolve(magnitude_spectrum_3, np.ones(smoothing_window) / smoothing_window,
                                      mode='valid')  # сглаживание

    diff_1 = smoothed_spectrum_1 - smoothed_spectrum_2
    diff_2 = smoothed_spectrum_1 - smoothed_spectrum_3
    diff_3 = smoothed_spectrum_2 - smoothed_spectrum_3

    if GlobalParams.show_spectrum == True:
        GlobalParams.show_spectrum = False
        print("Обрезанные спектры:")
        plt.figure(figsize=(15, 6))
        plt.plot(magnitude_spectrum_1, label="спектр 1", color='red')
        #plt.legend()
        plt.show()
        plt.figure(figsize=(15, 6))
        plt.plot(magnitude_spectrum_2, label="спектр 2", color='green')
        #plt.legend()
        plt.show()
        plt.figure(figsize=(15, 6))
        plt.plot(magnitude_spectrum_3, label="спектр 3", color='blue')
        #plt.legend()
        plt.show()

    return smoothed_spectrum_1, smoothed_spectrum_2, smoothed_spectrum_3, diff_1, diff_2, diff_3


def data_to_dataset(data):
    df_dataset = data.copy()

    # Создаём DataFrame из списка данных
    new_columns_df = pd.DataFrame([x for x in data["data"]], columns=[f'data_{i}' for i in range(len(data['data'][0]))])
    # Объединяем новые столбцы с основным DataFrame
    df_dataset = pd.concat([df_dataset, new_columns_df], axis=1)
    df_dataset.drop(columns=['data'], inplace=True)

    # Преобразуем целевую переменную в int
    df_dataset['label'] = df_dataset['label'].astype(int)

    return df_dataset


def data_to_tensor_dataset(x, y):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = TensorDataset(x_tensor, y_tensor)
    return dataset

