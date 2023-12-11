import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import variance


def load_data(filename, frame_size=100):
    # 1) отсечь начало
    # 2) оперделить размер фрейма
    # 3) выделить фрейм
    # 4) посчитать дисперсию
    # 5) сравнить с минимальной
    # 5.1) если размер дисперсии меньше минимальной, то текущая дисперсия = минимальная и вернуться на п.2
    # 6) сравнить с предыдущей
    # 6.1) если текущая дисперсия < предыдущей умножаем размер след. фрейма в 0.6 раз иначе умножаем в 1.2 раза
    # 7) предыдущая дисперсия = текущая дисперсия
    # 8) добавить фрейм к итоговому списку
    # 9) повторять п.2-9 до конца файла

    channel_1 = []
    channel_2 = []
    channel_3 = []

    # 1) отсечь начало
    # 2) оперделить размер фрейма
    min_variance_ch_1 = 1000
    min_variance_ch_2 = 1000
    min_variance_ch_3 = 1000
    previous_variance_ch_1 = 0
    previous_variance_ch_2 = 0
    previous_variance_ch_3 = 0
    is_end_file = False
    with open(filename) as file:
        for _ in range(21):  # скип ненужных строк
            next(file)

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
            # 5.1) если размер дисперсии меньше минимальной, то текущая дисперсия = минимальная и вернуться на п.2
            if is_low_variance(current_variance_ch_1, min_variance_ch_1, current_variance_ch_2, min_variance_ch_2, current_variance_ch_3, min_variance_ch_3):
                min_variance_ch_1 = current_variance_ch_1
                min_variance_ch_2 = current_variance_ch_2
                min_variance_ch_3 = current_variance_ch_3
                continue

            # 6) сравнить с предыдущей
            # 6.1) если текущая дисперсия < предыдущей умножаем размер след. фрейма в 1.2 раз иначе умножаем в 0.8 раз
            if is_low_variance(current_variance_ch_1, previous_variance_ch_1, current_variance_ch_2, previous_variance_ch_2, current_variance_ch_3, previous_variance_ch_3):
                frame_size = frame_size * 1.2
            else:
                frame_size = frame_size * 0.8
            frame_size = int(frame_size)

            # 7) предыдущая дисперсия = текущая дисперсия
            previous_variance_ch_1 = current_variance_ch_1
            previous_variance_ch_2 = current_variance_ch_2
            previous_variance_ch_3 = current_variance_ch_3

            # 8) добавить фрейм к итоговому списку
            channel_1 += ch_1_lst
            channel_2 += ch_2_lst
            channel_3 += ch_3_lst
            # 9) повторять п.2-9 до конца файла

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


def split_data_to_dataframe(channel_1, channel_2, channel_3, label, count_of_data_in_channel):
    lst_ch_1 = []
    lst_ch_2 = []
    lst_ch_3 = []
    norm_dataset = pd.DataFrame()
    diff_dataset = pd.DataFrame()

    for i, (ch_1, ch_2, ch_3) in enumerate(zip(channel_1, channel_2, channel_3)):
        lst_ch_1.append(ch_1)
        lst_ch_2.append(ch_2)
        lst_ch_3.append(ch_3)
        if i != 0 and (i + 1) % count_of_data_in_channel == 0:
            diff_1 = []
            diff_2 = []
            diff_3 = []
            for j in range(count_of_data_in_channel):
                diff_1.append(lst_ch_1[j] - lst_ch_2[j])
                diff_2.append(lst_ch_1[j] - lst_ch_2[j])
                diff_3.append(lst_ch_2[j] - lst_ch_3[j])

            norm_lst = lst_ch_1 + lst_ch_2 + lst_ch_3
            diff_lst = diff_1 + diff_2 + diff_3

            norm_dataset = pd.concat([norm_dataset, pd.DataFrame({"data": [norm_lst], "label": [label]})],
                                     ignore_index=True)
            diff_dataset = pd.concat([diff_dataset, pd.DataFrame({"data": [diff_lst], "label": [label]})],
                                     ignore_index=True)
            lst_ch_1 = []
            lst_ch_2 = []
            lst_ch_3 = []

    return norm_dataset, diff_dataset


def train_val_test_split(dataset):
    """
    Разделение на 3 выборки (обязательно на 3)
    :param dataset:
    :return:
    """
    labels = dataset['label']
    x_train, x_temp = train_test_split(dataset, test_size=0.2, stratify=labels, random_state=42)

    label_val_test = x_temp['label']
    x_valid, x_test = train_test_split(x_temp, test_size=0.5, stratify=label_val_test, random_state=42)

    return x_train, x_valid, x_test
