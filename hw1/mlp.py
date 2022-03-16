import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

# input and output dimensions
input_dim = 784
output_dim = 10


def relu(x):
    return (abs(x) + x) / 2


def softmax(x):
    exp = np.exp(x)
    b = np.sum(x, axis=1)
    c = b.reshape(b.shape[0], 1)
    return exp / c


def cross_entropy_loss(gt, y):
    log_y = np.log2(y)
    batch_size = gt.shape[0]
    loss = gt * log_y
    loss = -np.sum(loss) / batch_size
    return loss


def calc_loss_and_grad(x, y, w1, b1, w2, b2, eval_only=False):
    """Forward Propagation and Backward Propagation.

    Given a mini-batch of images x, associated labels y, and a set of parameters, compute the
    cross-entropy loss and gradients corresponding to these parameters.

    :param x: images of one mini-batch.
    :param y: labels of one mini-batch.
    :param w1: weight parameters of layer 1.
    :param b1: bias parameters of layer 1.
    :param w2: weight parameters of layer 2.
    :param b2: bias parameters of layer 2.
    :param eval_only: if True, only return the loss and predictions of the MLP.
    :return: a tuple of (loss, db2, dw2, db1, dw1)
    """

    # forward pass
    batch_size = y.shape[0]
    loss, y_hat = None, None
    z = x.dot(w1) + b1
    y_middle = relu(z)
    y_hat = softmax(y_middle.dot(w2) + b2)
    loss = cross_entropy_loss(y, y_hat)

    if eval_only:
        return loss, y_hat

    # TODO
    # backward pass
    db2 = (y_hat - y) / batch_size
    db2 = np.zeros_like(b2)
    for i in range(batch_size):
        db2 += y_hat[i] - y[i]
    db2 = db2 / batch_size
    dw2 = np.sum(y_hat - y) / batch_size * y_middle.T
    dw2 = np.zeros_like(w2)
    for i in range(batch_size):
        dw2 += (y_hat[i] - y[i]) * y_middle[i].T
    dw2 = dw2 / batch_size
    # left = (np.sum(y_hat-y) / batch_size * w2.T)
    # right = x.T.dot(np.sign(z))
    # dw1 = (np.sum(y_hat-y) / batch_size * w2.T) * (np.sign(z).dot(x))
    # db1 = (np.sum(y_hat-y) / batch_size * w2.T) * np.sign(z)

    return loss, db2, dw2 #, db1, dw1


def train(train_x, train_y, test_x, text_y, args: argparse.Namespace):
    """Train the network.

    :param train_x: images of the training set.
    :param train_y: labels of the training set.
    :param test_x: images of the test set.
    :param text_y: labels of the test set.
    :param args: a dict of hyper-parameters.
    """

    # TODO
    #  randomly initialize the parameters (weights and biases)
    w1, b1, w2, b2 = None, None, None, None

    print('Start training:')
    print_freq = 100
    loss_curve = []

    for epoch in range(args.epochs):
        # train for one epoch
        print("[Epoch #{}]".format(epoch))

        # random shuffle dataset
        dataset = np.hstack((train_x, train_y))
        np.random.shuffle(dataset)
        train_x = dataset[:, :input_dim]
        train_y = dataset[:, input_dim:]

        n_iterations = train_x.shape[0] // args.batch_size

        for i in range(n_iterations):
            # load a mini-batch
            x_batch = train_x[i * args.batch_size: (i + 1) * args.batch_size, :]
            y_batch = train_y[i * args.batch_size: (i + 1) * args.batch_size, :]

            # TODO
            # compute loss and gradients
            loss = None

            # TODO
            # update parameters

            loss_curve.append(loss)
            if i % print_freq == 0:
                print('[Iteration #{}/{}] [Loss #{:4f}]'.format(i, n_iterations, loss))

    # show learning curve
    plt.title('Training Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(loss_curve)), loss_curve)
    plt.show()

    # evaluate on the training set
    loss, y_hat = calc_loss_and_grad(train_x, train_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(train_y, axis=1)
    accuracy = np.sum(predictions == labels) / train_x.shape[0]
    print('Top-1 accuracy on the training set', accuracy)

    # evaluate on the test set
    loss, y_hat = calc_loss_and_grad(test_x, text_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(text_y, axis=1)
    accuracy = np.sum(predictions == labels) / test_x.shape[0]
    print('Top-1 accuracy on the test set', accuracy)


def main(args: argparse.Namespace):
    # print hyper-parameters
    print('Hyper-parameters:')
    print(args)

    # load training set and test set
    # train_x, train_y = utils.load_data("train")
    # test_x, text_y = utils.load_data("test")
    # print('Dataset information:')
    # print("training set size: {}".format(len(train_x)))
    # print("test set size: {}".format(len(test_x)))

    # check your implementation of backward propagation before starting training
    utils.check_grad(calc_loss_and_grad)

    # train the network and report the accuracy on the training and the test set
    # train(train_x, train_y, test_x, text_y, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multilayer Perceptron')
    parser.add_argument('--hidden-dim', default=50, type=int,
                        help='hidden dimension of the Multilayer Perceptron')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    args = parser.parse_args()
    main(args)
