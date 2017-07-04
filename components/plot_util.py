import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


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
    plt.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0)


def convert_bw(img_path):
    from PIL import Image
    image = Image.open(img_path).convert("L")
    new_path = "bw" + str(img_path)
    image.save(new_path)


class Plotter:
    """ Plots a variable amount of images """
    def __init__(self, cmap='gray'):
        self.cmap = cmap
        self.images = []
        self.plt = plt

    def add_to_plot(self, image, image_title):
        self.images.append([image, image_title])

    def save_plot(self, filename="output.png"):
        n_plots = len(self.images)
        # Smallest square that fits all n_plots
        figsize = [int(np.ceil(n_plots**(1/2)))]*2

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(figsize[0], figsize[1])
        gs.update(wspace=0.05, hspace=0.05)

        for i, (img, img_title) in enumerate(self.images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.set_title(img_title)
            plt.imshow(img, cmap='gray')

        self.plt.savefig(filename, bbox_inches='tight', transparent=True,
                         pad_inches=0)
        plt.close()
