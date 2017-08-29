# pylint: disable=all
import numpy as np
from scipy.stats import mode

def flip(bit):
    if int(bit) == int(1):
        return 0
    else:
        return 1

def add_noise(bits, pr):
    bits_w_noise = []
    for b in bits:
        if np.random.random() > pr:
            bits_w_noise.append(b)
        else:
            bits_w_noise.append(flip(b))
    return bits_w_noise
        
class RepetitionCode:
    def __init__(self, n_repeats=3):
        self.n = n_repeats

    def encode(self, bits):
        """ For each bit in the sequence, repeat it `n` times """
        return np.array([[b]*self.n for b in bits]).flatten()

    def decode(self, bits):
        # Split into buckets of size `n`
        buckets = [bits[i*self.n:(i+1)*self.n] for i in range(int(len(bits)/self.n))]
        # Most common value represents the entire bucket
        return mode(buckets, axis=1).mode.flatten() 


class Hamming74Code:
    """ Add 3 parity bits for a 4 sequence code
    
    Works for any single flip, but if there's more than 1 flip it fails
    """
    def __init__(self):
        pass

    def encode(self, bits):
        r1, r2, r3, r4 = bits
        r5 = (r1 + r2 + r3) % 2
        r6 = (r2 + r3 + r4) % 2
        r7 = (r1 + r3 + r4) % 2
        return [r1, r2, r3, r4, r5, r6, r7]

    def decode(self, bits):
        r1, r2, r3, r4, r5, r6, r7 = bits
        # Get the sum of elements, parity is 0 if even, 1 if odd
        r5_parity = (r1 + r2 + r3) % 2 == r5
        r6_parity = (r2 + r3 + r4) % 2 == r6
        r7_parity = (r1 + r3 + r4) % 2 == r7

        if not r5_parity and r6_parity and r7_parity:
            r1 = flip(r1)
        elif not r5_parity and not r6_parity and r7_parity:
            r2 = flip(r2)
        elif not r5_parity and not r6_parity and not r7_parity:
            r3 = flip(r3)
        elif r5_parity and not r6_parity and not r7_parity:
            r4 = flip(r4)
        return [r1, r2, r3, r4, r5, r6, r7]

def capacity_bsc(f):
    """ Calculates Optimal Error Rate for Binary Symmetric Channel """
    return 1-(f*np.log2(1/f)+(1-f)*np.log2(1/(1-f)))

if __name__ == "__main__":
    # Too lazy to make actual tests, just going to print everything

    print("====== PARAMS ======")
    # Size of bit sequnece
    LENGTH = 10
    print("Number of bits in sequence: {}".format(LENGTH))
    # Amount of noise added/probability of a bit flipping
    ERROR_RATE = 0.2
    print("Probability of Noise: {}".format(ERROR_RATE))

    print("")
    print("====== Repetition Code ======")
    bits = np.random.choice([0, 1], size=(LENGTH))
    print("Original : {}".format(bits))
    system = RepetitionCode(n_repeats=5)
    encoded_bits = system.encode(bits)
    print("Encoded: {}".format(encoded_bits))
    encoded_bits_w_noise = add_noise(encoded_bits, ERROR_RATE)
    print("Noise Added (pr={}): {}".format(ERROR_RATE, encoded_bits_w_noise))
    decoded_bits = system.decode(encoded_bits_w_noise)
    print("Decoded: {}".format(decoded_bits))
    print("# of Errors: {}".format(np.sum(np.abs(bits - decoded_bits))))

    print("")
    print("====== Hamming 7-4 Code ======")
    # bits = np.random.choice([0, 1], size=(4))
    bits = np.array([1, 0, 0, 0])
    print("Original : {}".format(bits))
    system = Hamming74Code()
    encoded_bits = system.encode(bits)
    print("Encoded: {}".format(encoded_bits))
    encoded_bits_w_noise = add_noise(encoded_bits, ERROR_RATE)
    print("Noise Added (pr={}): {}".format(ERROR_RATE, encoded_bits_w_noise))
    decoded_bits = system.decode(encoded_bits_w_noise)
    print("Decoded: {}".format(decoded_bits))
    print("# of Errors: {}".format(np.sum(np.abs(bits - decoded_bits[:4]))))
