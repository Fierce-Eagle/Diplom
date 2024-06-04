from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from dataset_processing import *
from signal_processing import *
from global_params import *
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import os

def create_dataset(dir_name=None, time_for_frame=2, frame_time_coeff=4, frame_stride_percent=0.25,
                   smoothing_window_size=10, spectrum_length=500, batch_size=32,
                   is_regression_dataset=False):
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
    train_dataset = pd.DataFrame()
    valid_dataset = pd.DataFrame()

    GlobalParams.show_spectrum = False  #!!!!!!!!!!!!!!!!!!!!!!!!!
    GlobalParams.show_signal = False
    for num_class, folder_name in enumerate(os.listdir(dir_name)):
        #if num_class > 0 and num_class < 4:
        #    print(num_class)
        #    continue
        if is_regression_dataset:
            label = int(folder_name)
        else:
            if num_class == 0:
                label = 0
                stride = frame_stride_percent * 1.2
            else:
                label = 1
                stride = frame_stride_percent

        folder_path = os.path.join(dir_name, folder_name) + '/'
        # Проходим по файлам с данными
        for file_name in os.listdir(folder_path):
            filename = os.path.join(folder_path, file_name)
            filenum = int(filename.split('о')[1].split('.')[0])

            #if filenum == 10:
            #    continue

            signal_1, signal_2, signal_3, frequency = load_info_from_file(filename)

            if GlobalParams.show_signal:
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

            # обработка сигналов
            signal_1, signal_2, signal_3 = cut_empty_frames_in_signals(signal_1, signal_2, signal_3, frequency)

            if GlobalParams.show_signal:
                GlobalParams.show_signal = False
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

            if label == 0:
                coeff = 1
            else:
                coeff = frame_time_coeff

            frame_size = int(time_for_frame * frequency * coeff)
            frame_stride = int(stride * frame_size)
            # создание спектров
            dataset_path = split_data_to_dataframe(signal_1, signal_2, signal_3, label,
                                                   frame_size,
                                                   frame_stride,
                                                   coeff,
                                                   smoothing_window_size,
                                                   spectrum_length)

            if filenum <= 8:
                train_dataset = pd.concat([train_dataset, dataset_path], ignore_index=True)
            else:
                valid_dataset = pd.concat([valid_dataset, dataset_path], ignore_index=True)


    train_data = data_to_dataset(train_dataset)
    valid_data = data_to_dataset(valid_dataset)

    x_train = train_data.drop('label', axis=1)
    y_train = train_data['label']

    x_valid = valid_data.drop('label', axis=1)
    y_valid = valid_data['label']

    # Нормализация
    scaler = MaxAbsScaler().fit(x_train)
    GlobalParams.scaler = scaler

    x_train_norm = scaler.transform(x_train)
    x_valid_norm = scaler.transform(x_valid)

    # Преобразование в тензоры
    train_dataset = data_to_tensor_dataset(x_train_norm, y_train)
    valid_dataset = data_to_tensor_dataset(x_valid_norm, y_valid)

    if not is_regression_dataset:
        train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataset = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_dataset = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    return train_dataset, valid_dataset


def create_test_dataset(folder_path=None, time_for_frame=2, frame_stride_percent=0.25, smoothing_window_size=10,
                        spectrum_length=500, batch_size=4):
    final_dataset = pd.DataFrame()

    GlobalParams.show_spectrum = False  # !!!!!!!!!!!!!!!!!!!!!!!!!

    for file_name in os.listdir(folder_path):
        filename = os.path.join(folder_path, file_name)
        filenum = int(filename.split('о')[1].split('.')[0])

        #if filenum < 10:
        #    continue
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
    #print(x_test.values.shape)

    # Нормализация
    x_test_norm = GlobalParams.scaler.transform(x_test)
    #print(len(x_test_norm[0]))
    # Преобразование в тензоры
    test_dataset = data_to_tensor_dataset(x_test_norm, y_test)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset
