import matplotlib.pyplot as plt


def plot_img_before_after(image1, image2, filename="out.png", cmap="gray"):
    """ Plot 2 images, side by side"""
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(image1, cmap=cmap)
    ax1.set_title("Before")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(image2, cmap=cmap)
    ax2.set_title("After")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.axis('off')
    plt.savefig(filename)
