""" components.plot_util.py"""
import os
import unittest
import numpy as np
from models.components.plot_util import Plotter

# pylint: disable=missing-docstring
class TestPlotUtil(unittest.TestCase):
    """
    Tests
    -----
    image_added: Check that an image is added to the list of plots
    filed_created: Check that the image was saved to disk

    """
    def setUp(self):
        self.img = np.random.randn(100, 100)
        self.path = "test_output.png"

    def tearDown(self):
        # Delete generated image
        if self.path in os.listdir():
            os.remove(self.path)

    def test_plot_added(self):
        plt = Plotter()
        plt.add_to_plot(self.img, "image")
        self.assertEqual(len(plt.images), 1)

    def test_filed_created(self):
        plt = Plotter()
        for i in range(15):
            plt.add_to_plot(self.img, "image-{}".format(i))
        plt.save_plot(self.path)
        self.assertTrue(self.path in os.listdir())

if __name__ == "__main__":
    unittest.main()
