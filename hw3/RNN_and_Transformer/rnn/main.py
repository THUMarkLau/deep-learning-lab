# coding: utf-8
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from visdom import Visdom

import data
import model

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')
parser.add_argument('--vis', default='test')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--s', type=int, default=3)

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
# Use gpu or cpu to train
use_gpu = True
args.gpu_id = 0
vis = Visdom(env=args.vis)
if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)

# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)
nvoc = len(data_loader.vocabulary)
net = model.LMModel(nvoc, 400, 256, 2)
if use_gpu:
    net = net.cuda(device)
net.init_weights()
print(net)
########################################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(net, data_loader, criterion):
    net.train(False)
    total_loss = 0.0
    # total_correct = 0
    data_loader.set_valid()
    data, target, end_flag = data_loader.get_batch()
    data, target = data.to(device), target.to(device)
    while not end_flag:
        output, hidden = net(data)
        output = output.view(output.size(0) * output.size(1), output.size(2))
        loss = criterion(output, target)
        # _, predictions = torch.max(output, 1)
        total_loss += loss.item() * data.size(0)
        data, target, end_flag = data_loader.get_batch()
        data, target = data.to(device), target.to(device)
    epoch_loss = total_loss / data_loader.valid.shape[0]
    return epoch_loss


########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train(net, data_loader, optimizer, criterion):
    net.train(True)
    total_loss = 0.0
    # total_correct = 0
    data_loader.set_train()
    data, target, end_flag = data_loader.get_batch()
    data, target = data.to(device), target.to(device)
    while not end_flag:
        net.zero_grad()
        output, hidden = net(data)
        output = output.view(output.size(0) * output.size(1), output.size(2))
        loss = criterion(output, target)
        # _, predictions = torch.max(output, 1)
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        data, target, end_flag = data_loader.get_batch()
        data, target = data.to(device), target.to(device)
    epoch_loss = total_loss / data_loader.train.shape[0]
    # epoch_acc = total_correct.double() /  (data_loader.train.shape[0] * data_loader.train.shape[1])

    return epoch_loss  # , epoch_acc.item()


########################################


# Loop over epochs.
best_avg_loss = 10000
train_avg_loss = []
valid_avg_loss = []
train_win = None
valid_win = None
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [(args.epochs * i) // args.s for i in range(1, args.s + 1)],
                                                 gamma=0.3,
                                                 last_epoch=-1)
print(scheduler)
for epoch in tqdm(range(1, args.epochs + 1)):
    print('epoch:{:d}/{:d}'.format(epoch, args.epochs))
    print('*' * 100)
    train_loss = train(net, data_loader, optimizer, criterion)
    print("training: {:.4f}".format(train_loss))
    valid_loss = evaluate(net, data_loader, criterion)
    if train_win is None:
        train_win = vis.line(X=np.array([epoch]), Y=np.array([np.exp(train_loss)]),
                             opts=dict(title="loss", xlabel="epoch", showlegend=True), name='train_loss')
    else:
        vis.line(X=np.array([epoch]), Y=np.array([np.exp(train_loss)]), win=train_win, update='append',
                 name='train_loss')
    vis.line(X=np.array([epoch]), Y=np.array([np.exp(valid_loss)]), win=train_win, update='append', name='valid_loss')
    scheduler.step()
    print("validation: {:.4f}".format(valid_loss))
    train_avg_loss.append(train_loss)
    valid_avg_loss.append(valid_loss)
    if valid_loss < best_avg_loss:
        best_avg_loss = valid_loss
        best_model = net
        torch.save(best_model, 'best_model.pt')

perplexity_train = torch.exp(torch.Tensor(train_avg_loss))
perplexity_valid = torch.exp(torch.Tensor(valid_avg_loss))
with open("PPL_" + args.vis + ".txt", "w") as f:
    f.write("PPL Train: " + str(perplexity_train))
    f.write("\n")
    f.write("PPL Valid: " + str(perplexity_valid))
    f.write("\n")
print("PPL Train: ", perplexity_train)
print("PPL Valid: ", perplexity_valid)
