from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataset_processing import *
from signal_processing import *
from matplotlib import pyplot as plt
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch
import os


def create_dataset(dir_name=None, signal_start_frame_size_to_read=5000, signal_frame_size_to_read=200,
                   time_for_frame=2, frame_stride_percent=0.25,
                   smoothing_window_size=10, spectrum_length=500, train_size=0.6, batch_size=32,
                   is_regression_dataset=False, show_info=False):
    """
    Создание датасета
    @param dir_name: корневая папка
    @param signal_start_frame_size_to_read: размер начального кадра с выбросами при считывании сигнала из файла.
    @param signal_frame_size_to_read: размер кадра который будет сравниваться с эталоном выбросов при считывании сигнала
    из файла.
    @param time_for_frame: время в секундах для создания фрейма.
    @param frame_stride_percent: размер отступа от размера фрейма от 0 до 1.
    @param smoothing_window_size: размер окна сглаживания спектра.
    @param spectrum_length: длина спектра. ПРИМЕЧАНИЕ: фрейм состоит из 6 спектров.
    @param train_size: размер тренировочной выборки от 0 до 1.
    @param batch_size: размер батча.
    @param is_regression_dataset: влияет на метку во фрейме. Использовать значение True при решении задачи регрессии.
    @param show_info: отображение информации.
    @return:
    """
    # Проходим по папкам-классам
    final_dataset = pd.DataFrame()
    for num_class, folder_name in enumerate(os.listdir(dir_name)):
        if is_regression_dataset:
            label = int(folder_name)
        else:
            label = num_class

        folder_path = os.path.join(dir_name, folder_name) + '/'
        # Проходим по файлам с данными
        for file_name in os.listdir(folder_path):
            filename = os.path.join(folder_path, file_name)
            print(filename)
            # считывание сигналов
            signal_1, signal_2, signal_3, freqence = load_info_from_file(filename, show_info)

            # обработка сигналов
            signal_1, signal_2, signal_3 = cut_empty_frames_in_signals(signal_1, signal_2, signal_3)

            frame_size = int(time_for_frame * freqence)
            frame_stride = int(frame_stride_percent * frame_size)
            # создание спектров
            dataset_path = split_data_to_dataframe(signal_1, signal_2, signal_3, label,
                                                                      frame_size,
                                                                      frame_stride,
                                                                      smoothing_window_size,
                                                                      spectrum_length)

            final_dataset = pd.concat([final_dataset, dataset_path], ignore_index=True)

    final_data = data_to_dataset(final_dataset)
    # Выделение данных
    x_train, x_valid, x_test, y_train, y_valid, y_test = train_val_test_split(final_data, train_size=train_size)

    # Нормализация
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_valid_norm = scaler.transform(x_valid)
    x_test_norm = scaler.transform(x_test)

    # Преобразование в тензоры
    train_dataset = data_to_tensor_dataset(x_train_norm, y_train)
    valid_dataset = data_to_tensor_dataset(x_valid_norm, y_valid)
    test_dataset = data_to_tensor_dataset(x_test_norm, y_test)

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, valid_dataset, test_dataset





def train_val_test_split(dataset, train_size=0.6):
    """
    Разделение на 3 выборки (обязательно на 3)
    @param dataset: датасет вида pd.DataFrame({"data от 0 до n": [data(0-n)], "label": [label]})
    @param train_size: размер тренировочной выборки от 0 до 1
    @return:
    """
    data = dataset.drop('label', axis=1)
    labels = dataset['label']
    x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=(1 - train_size), random_state=42)

    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    return x_train, x_valid, x_test, y_train, y_valid, y_test




