from os import read
import re
import subprocess

import numpy as np
import matplotlib.image as mpimg

# Stole this function from
# http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm/7369986#7369986


class PgmProcesser:
    def __init__(self):
        pass

    @classmethod
    def read_pgm(cls, filename, byteorder='>'):
        """Return image data from a raw PGM file as numpy array.
        Format specification: http://netpbm.sourceforge.net/doc/pgm.html
        """
        with open(filename, 'rb') as f:
            buffer_ = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer_).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)
        raw_array = np.frombuffer(buffer_,
                                  dtype='u1' if int(
                                      maxval) < 256 else byteorder+'u2',
                                  count=int(width)*int(height),
                                  offset=len(header)
                                  ).reshape((int(height), int(width)))
        return np.where(raw_array > 0, 0, 1)
