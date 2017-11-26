from matplotlib import pyplot as plt
import numpy as np


def plot_imgs(inputs):
    assert len(inputs[0]) == 96**2
    """Plot smallNORB images helper"""
    fig = plt.figure()
    plt.title('Show images')
    r = np.floor(np.sqrt(len(inputs))).astype(int)
    for i in range(r**2):
        sample = inputs[i, :].reshape(96, 96)
        a = fig.add_subplot(r, r, i + 1)
        a.imshow(sample, cmap='gray')
    plt.show()
