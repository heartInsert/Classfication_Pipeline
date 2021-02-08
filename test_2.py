from sklearn.metrics import confusion_matrix
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    target = np.array([1, 1, 0, 1, 2])
    pred = np.array([0, 1, 0, 1, 0])
    result = confusion_matrix(target, pred)
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    plot = skplt.metrics.plot_confusion_matrix(target, pred, normalize=False, ax=axes)
    fig.savefig('save_img3.jpg')
    fig, axes = plt.subplots(1, 1, figsize=None)
    plot = skplt.metrics.plot_confusion_matrix(target, pred, normalize=True, ax=axes)
    fig.savefig('save_img4.jpg', dpi=400)
    # plt.show()
    print()
