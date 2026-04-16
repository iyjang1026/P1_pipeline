import astropy.io.fits as fits
import pyimfit
from photutils.segmentation import make_2dgaussian_kernel

path = '/volumes/ssd/'

hdu = fits.getdata(path)
hdr = fits.getheader(path)

mask = fits.getdata(path)
psf = make_2dgaussian_kernel(5,15)

