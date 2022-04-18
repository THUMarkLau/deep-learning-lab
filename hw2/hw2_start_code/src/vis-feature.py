import models
import sys
import data
import torch as t
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.image as pimg

if __name__ == '__main__':
    model = t.load(sys.argv[1] if len(sys.argv) > 1 else "best_model-A-RMSprop.pt")
    model.eval()
    model.to("cuda:0")

    features = []
    labels = []

    hook = lambda _, x, __ : features.append(x[0].detach().cpu())
    register_hook = model.fc.register_forward_hook(hook)
    _, test_data = data.load_data()
    with t.no_grad():
        for _, (data, label) in enumerate(test_data):
            data = data.to('cuda:0')
            model(data)
            labels.append(label)

    register_hook.remove()
    features = t.cat([x for x in features]).numpy()
    labels = t.cat([x for x in labels]).numpy()


    def visualize_tsne(features, targets):
        X = TSNE(n_components=2, random_state=0).fit_transform(features)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        ax.set_title('T-SNE Visualization')
        sequence = ax.scatter(X[:, 0], X[:, 1], c=targets, s=20)
        plt.show()

    visualize_tsne(features, labels)

