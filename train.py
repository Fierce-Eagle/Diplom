import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_model(loader_train, loader_val, cnn_model, epochs=10, device=None, lr=1e-3, criterion=nn.CrossEntropyLoss()):
    """
    return: потери +, лучшая модель +, предсказанные оценки для валидационного набора
    """
    assert device is not None, "device must be cpu or cuda"
    optimizer = optim.Adagrad(cnn_model.parameters(), lr, weight_decay=5e-8)
    loss_history = []  # потери
    model = cnn_model.to(device)
    best_model = None  # лучшая модель
    best_acc = 0
    best_loss = 10000
    batch_num = len(loader_train)

    y_true_valid = []
    for (_, labels) in loader_val:
        y_true_valid += [int(y.item()) for y in labels]
    for i, _ in enumerate(y_true_valid):
        if y_true_valid[i] != 0:
            y_true_valid[i] = 1

    y_true_valid = torch.Tensor(y_true_valid)

    for epoch in range(epochs):
        loss_sum = 0
        model.train()
        for (x, y) in loader_train:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.int64)

            y = torch.flatten(y)

            for i, _ in enumerate(y):
                if y[i] != 0:
                    y[i] = 1

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
        if current_acc >= best_acc:
            if current_loss < best_loss:
                best_loss = current_loss
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


def train_regression(train_dataloader, reg_model, epochs=10, device=None, lr=1e-3,
                     criterion=nn.MSELoss(size_average=False)):
    """
    return: потери +, лучшая модель +, предсказанные оценки для валидационного набора
    """
    assert device is not None, "device must be cpu or cuda"
    optimizer = optim.Adam(reg_model.parameters(), lr)
    loss_history = []  # потери
    model = reg_model.to(device)

    for epoch in range(epochs):
        loss_sum = 0

        for batch, (x, y) in enumerate(train_dataloader):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_loss = loss_sum / (batch + 1)
        loss_history.append(current_loss)
        print('Epoch [%d/%d], loss = %.4f' % (epoch, epochs, current_loss))

    return loss_history, model

