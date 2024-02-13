from sklearn.model_selection import train_test_split
from statistics import variance
import pandas as pd


def load_data(filename):
    """
    Загрузка данных с отсечением начала и конца динамически
    @param filename:
    @return:
    """
    channel_1 = []
    channel_2 = []
    channel_3 = []
    ch_1_lst = []
    ch_2_lst = []
    ch_3_lst = []
    frame_size = 2000

    # 1) отсечь начало
    # 2) оперделить размер фрейма
    is_end_file = False
    with open(filename) as file:
        for _ in range(21):  # скип ненужных строк
            next(file)

        for _ in range(5000):  # определение минимальной дисперсии
            line = next(file)
            channel_params = line.split()
            ch_1_lst.append(float(channel_params[1]))
            ch_2_lst.append(float(channel_params[2]))
            ch_3_lst.append(float(channel_params[3]))

        min_variance_ch_1 = variance(ch_1_lst) * 1.1
        min_variance_ch_2 = variance(ch_2_lst) * 1.1
        min_variance_ch_3 = variance(ch_3_lst) * 1.1

        while not is_end_file:
            # 3) выделить фрейм
            ch_1_lst = []
            ch_2_lst = []
            ch_3_lst = []
            for _ in range(frame_size):
                line = next(file, 100)
                if line == 100:
                    is_end_file = True
                    break
                channel_params = line.split()  # получение значений для каналов
                ch_1_lst.append(float(channel_params[1]))
                ch_2_lst.append(float(channel_params[2]))
                ch_3_lst.append(float(channel_params[3]))

            if len(ch_1_lst) < 2:
                break
            # 4) посчитать дисперсию
            current_variance_ch_1 = variance(ch_1_lst)
            current_variance_ch_2 = variance(ch_2_lst)
            current_variance_ch_3 = variance(ch_3_lst)

            # 5) сравнить с минимальной
            # 5.1) если размер дисперсии меньше минимальной, то вернуться на п.3
            if not is_low_variance(current_variance_ch_1, min_variance_ch_1, current_variance_ch_2, min_variance_ch_2,
                                   current_variance_ch_3, min_variance_ch_3):
                continue

            # 6) добавить фрейм к итоговому списку
            channel_1 += ch_1_lst
            channel_2 += ch_2_lst
            channel_3 += ch_3_lst
            # 7) повторять п.3-7 до конца файла

    return channel_1, channel_2, channel_3


def is_low_variance(cur_ch_1, prev_ch_1, cur_ch_2, prev_ch_2, cur_ch_3, prev_ch_3):
    count = 0
    if cur_ch_1 < prev_ch_1:
        count += 1
    if cur_ch_2 < prev_ch_2:
        count += 1
    if cur_ch_3 < prev_ch_3:
        count += 1

    if count >= 2:
        return False
    else:
        return True


def train_val_test_split(dataset):
    """
    Разделение на 3 выборки (обязательно на 3)
    :param dataset:
    :return:
    """
    labels = dataset['label']
    x_train, x_temp = train_test_split(dataset, test_size=0.4, stratify=labels, random_state=42)

    label_val_test = x_temp['label']
    x_valid, x_test = train_test_split(x_temp, test_size=0.5, stratify=label_val_test, random_state=42)

    return x_train, x_valid, x_test
