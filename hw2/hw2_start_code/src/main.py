import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
from visdom import Visdom
import numpy as np
import time
from tqdm import tqdm
from LDMALoss import LDAMLoss
from CBLoss import CBLoss

vis = Visdom(env='hw2-TaskC-49_20')
train_loss_win = None
valid_loss_win = None
train_acc_win = None
valid_acc_win = None
loss_type = "CB"

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20):
    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    global vis
    global train_loss_win
    global valid_loss_win
    global train_acc_win
    global valid_acc_win
    for epoch in tqdm(range(num_epochs)):
        start_time = time.time()
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        epoch_time = time.time() - start_time
        start_time = time.time()
        if train_loss_win is None:
            train_loss_win = vis.line(X=np.array([epoch]), Y=np.array([train_loss]), opts=dict(title='train_loss'))
            valid_loss_win = vis.line(X=np.array([epoch]), Y=np.array([valid_loss]), opts=dict(title='valid_loss'))
            train_acc_win = vis.line(X=np.array([epoch]), Y=np.array([train_acc]), opts=dict(title='train_acc'))
            valid_acc_win = vis.line(X=np.array([epoch]), Y=np.array([valid_acc]), opts=dict(title='valid_acc'))
        else:
            vis.line(X=np.array([epoch]), Y=np.array([train_loss]), win=train_loss_win, update='append')
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss]), win=valid_loss_win, update='append')
            vis.line(X=np.array([epoch]), Y=np.array([train_acc]), win=train_acc_win, update='append')
            vis.line(X=np.array([epoch]), Y=np.array([valid_acc]), win=valid_acc_win, update='append')
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model-C-4920.pt')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "../data/"  ## You need to specify the data_dir first
    input_size = 224
    batch_size = 32

    ## about training
    num_epochs = 500
    lr = 0.001

    ## model initialization
    model = models.model_A(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size, topic="4-Long-Tailed")
    # train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)
    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    ## loss function
    cls_num_dict = dict()
    for input, labels in train_loader:
        for l in labels:
            l = l.item()
            if l in cls_num_dict:
                cls_num_dict[l] = cls_num_dict[l] + 1
            else:
                cls_num_dict[l] = 1
    cls_num_list = []
    for i in range(0, 10):
        cls_num_list.append(cls_num_dict[i])
    if loss_type == "LDMA":
        criterion = LDAMLoss(np.array(cls_num_list))
    elif loss_type == "CB":
        criterion = CBLoss(cls_num_list, num_classes, loss_type="focal", beta=0.9999, gamma=2.0).to("cuda:0")
    else:
        criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)
