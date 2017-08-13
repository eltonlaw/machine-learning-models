""" test_undercomplete_ae_np.py"""
import unittest
import numpy as np
from models.autoencoder.undercomplete_ae.undercomplete_ae_np import UAE


class TestUAE(unittest.TestCase):
    """Test Implementation of Undercomplete Autoencoder """
    def setUp(self):
        pass

    @unittest.skip("function unfinished")
    def test_default_parms(self):
        """Check return type, should return an np.ndarray"""
        model = UAE()
        self.assertEqual(model.lr, 0.0005)
        self.assertEqual(model.batch_size, 256)

    @unittest.skip("function unfinished")
    def test_pass_custom_params(self):
        """Check return type, should return an np.ndarray"""
        params = {
            "lr":0.01
            }
        model = UAE(parameters=params)
        self.assertEqual(model.lr, 0.01)
        self.assertEqual(model.batch_size, 256)

if __name__ == "__main__":
    unittest.main()
