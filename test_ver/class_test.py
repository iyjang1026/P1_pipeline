import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import glob
import sys

class Master:
    def __init__(self, path, obj):
        self.path = path
        self.obj = obj
    
    def imp_arr(self):
        arr = fits.getdata(self.path)
        plt.imshow(arr, norm=simple_norm(arr, 'linear', percent=99), origin='lower')
        plt.show()
        self.arr = arr
    
    def median_comb(self):
        median = np.ma.median(self.arr)

Master('/volumes/ssd/test/M101.fits', 'M101').imp_arr()
