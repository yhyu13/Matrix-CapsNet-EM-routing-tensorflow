from matplotlib import pyplot as plt


def plot_imgs(inputs):
    """Plot smallNORB images helper"""
    fig = plt.figure()
    plt.title('Show images')
    r = np.floor(np.sqrt(len(inputs))).astype(int)
    for i in range(r**2):
        sample = images[i, :].reshape(96, 96)
        a = fig.add_subplot(r, r, i + 1)
        a.imshow(sample)
    plt.show()
