import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_model(loader_train, loader_val, cnn_model, epochs=10, device=None, lr=1e-3, criterion=nn.CrossEntropyLoss()):
    """
    return: потери +, лучшая модель +, предсказанные оценки для валидационного набора
    """
    assert device is not None, "device must be cpu or cuda"
    optimizer = optim.Adam(cnn_model.parameters(), lr)
    loss_history = []  # потери
    model = cnn_model.to(device)
    best_model = None  # лучшая модель
    best_acc = 0
    batch_num = len(loader_train)

    y_true_valid = []
    for (_, labels) in loader_val:
        y_true_valid += [float(y.item()) for y in labels]
    y_true_valid = torch.Tensor(y_true_valid)

    for epoch in range(epochs):
        loss_sum = 0
        model.train()
        for (x, y) in loader_train:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.int64)
            y = torch.flatten(y)

            optimizer.zero_grad()
            predicted_y = model(x)
            loss = criterion(predicted_y, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        current_loss = loss_sum / batch_num
        loss_history.append(current_loss)

        y_pred_valid = test_model(model, loader_val, device)
        y_pred_valid = torch.Tensor(y_pred_valid)

        correct = y_pred_valid.eq(y_true_valid)
        current_acc = torch.mean(correct.float())

        print('Epoch [%d/%d], loss = %.4f acc_val = %.4f' % (epoch, epochs, current_loss, current_acc))
        if current_acc > best_acc:
            best_acc = current_acc
            best_model = model

    print("Лучшая точность:", best_acc)

    return loss_history, best_model


def test_model(model, loader_test, device=None):
    assert device is not None, "device must be cpu or cuda"
    model.eval()
    predict_list = []

    with torch.no_grad():
        for x, _ in loader_test:
            x = x.to(device=device, dtype=torch.float32)
            scores = model(x).to(device)
            pred = torch.argmax(scores, dim=1)
            predict_list += [p.item() for p in pred]

    return np.array(predict_list)


def train_regression(loader_train, loader_val, cnn_model, epochs=10, device=None, lr=1e-3, criterion=nn.MSELoss()):
    """
    return: потери +, лучшая модель +, предсказанные оценки для валидационного набора
    """
    assert device is not None, "device must be cpu or cuda"
    optimizer = optim.Adam(cnn_model.parameters(), lr)
    loss_history = []  # потери
    model = cnn_model.to(device)
    best_model = None  # лучшая модель
    best_acc = 0
    batch_num = len(loader_train)

    y_true_valid = []
    for (_, labels) in loader_val:
        y_true_valid += [float(y.item()) for y in labels]
    y_true_valid = torch.Tensor(y_true_valid)

    for epoch in range(epochs):
        loss_sum = 0
        model.train()
        for (x, y) in loader_train:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y = torch.flatten(y)

            optimizer.zero_grad()
            predicted_y = model(x)
            loss = criterion(predicted_y, y.view(-1, 1))
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        current_loss = loss_sum / batch_num
        loss_history.append(current_loss)

        y_pred_valid = test_model(model, loader_val, device)
        y_pred_valid = torch.Tensor(y_pred_valid)

        correct = y_pred_valid.eq(y_true_valid)
        current_acc = torch.mean(correct.float())

        print('Epoch [%d/%d], loss = %.4f acc_val = %.4f' % (epoch, epochs, current_loss, current_acc))
        if current_acc > best_acc:
            best_acc = current_acc
            best_model = model

    print("Лучшая точность:", best_acc)

    return loss_history, best_model
