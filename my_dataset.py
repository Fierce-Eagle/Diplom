from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset_processing import *
from signal_processing import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import os

def create_dataset(dir_name=None, time_for_frame=2, frame_stride_percent=0.25,
                   smoothing_window_size=10, spectrum_length=500, train_size=0.6, batch_size=32,
                   is_regression_dataset=False, show_info=False):
    """
    Создание датасета
    @param dir_name: корневая папка
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
    show_spectrs = True
    for num_class, folder_name in enumerate(os.listdir(dir_name)):
        if is_regression_dataset:
            label = int(folder_name)
        else:
            label = num_class

        folder_path = os.path.join(dir_name, folder_name) + '/'
        # Проходим по файлам с данными
        for file_name in os.listdir(folder_path):
            filename = os.path.join(folder_path, file_name)
            # if show_info:
            #    print(filename)
            # считывание сигналов
            signal_1, signal_2, signal_3, frequency = load_info_from_file(filename)

            """
            if show_info:
                print("Частота:", frequency)
                plt.figure(figsize=(15, 6))
                plt.plot(signal_1, label="сигнал 1", color='red')
                plt.legend()
                plt.show()
                plt.figure(figsize=(15, 6))
                plt.plot(signal_2, label="сигнал 2", color='green')
                plt.legend()
                plt.show()
                plt.figure(figsize=(15, 6))
                plt.plot(signal_3, label="сигнал 3", color='blue')
                plt.legend()
                plt.show()
            """

            # обработка сигналов
            signal_1, signal_2, signal_3 = cut_empty_frames_in_signals(signal_1, signal_2, signal_3, frequency)

            """
            if show_info:
                print("Обрезанные сигналы:")
                plt.figure(figsize=(15, 6))
                plt.plot(signal_1, label="сигнал 1", color='red')
                plt.legend()
                plt.show()
                plt.figure(figsize=(15, 6))
                plt.plot(signal_2, label="сигнал 2", color='green')
                plt.legend()
                plt.show()
                plt.figure(figsize=(15, 6))
                plt.plot(signal_3, label="сигнал 3", color='blue')
                plt.legend()
                plt.show()
            """

            frame_size = int(time_for_frame * frequency)
            frame_stride = int(frame_stride_percent * frame_size)
            # создание спектров
            dataset_path = split_data_to_dataframe(signal_1, signal_2, signal_3, label,
                                                   frame_size,
                                                   frame_stride,
                                                   smoothing_window_size,
                                                   spectrum_length, show_spectrs)
            show_spectrs = False

            final_dataset = pd.concat([final_dataset, dataset_path], ignore_index=True)

    final_data = data_to_dataset(final_dataset)

    data = final_data.drop('label', axis=1)
    labels = final_data['label']

    # Выделение данных
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=(1 - train_size), random_state=42)

    # Нормализация
    scaler = StandardScaler().fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    # Преобразование в тензоры
    train_dataset = data_to_tensor_dataset(x_train_norm, y_train)
    test_dataset = data_to_tensor_dataset(x_test_norm, y_test)

    if not is_regression_dataset:
        train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, scaler


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


def create_test_dataset(folder_path=None, time_for_frame=2, frame_stride_percent=0.25, smoothing_window_size=10,
                        spectrum_length=500, batch_size=64, scaler=None):
    final_dataset = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        filename = os.path.join(folder_path, file_name)

        # считывание сигналов
        signal_1, signal_2, signal_3, frequency = load_info_from_file(filename)
        # обработка сигналов
        signal_1, signal_2, signal_3 = cut_empty_frames_in_signals(signal_1, signal_2, signal_3, frequency)
        frame_size = int(time_for_frame * frequency)
        frame_stride = int(frame_stride_percent * frame_size)
        # создание спектров
        dataset_path = split_data_to_dataframe(signal_1, signal_2, signal_3, label=1,
                                               frame_size=frame_size,
                                               frame_stride=frame_stride,
                                               spectrum_length=spectrum_length,
                                               smoothing_window=smoothing_window_size)

        final_dataset = pd.concat([final_dataset, dataset_path], ignore_index=True)

    final_data = data_to_dataset(final_dataset)
    x_test = final_data.drop('label', axis=1)
    y_test = final_data['label']
    # print(x_test)
    # Нормализация
    x_test_norm = scaler.transform(x_test)

    # Преобразование в тензоры
    test_dataset = data_to_tensor_dataset(x_test_norm, y_test)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset
