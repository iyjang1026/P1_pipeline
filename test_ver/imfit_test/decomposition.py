import numpy as np
from astropy.table import Table
from astropy.modeling.models import Sersic1D
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import astropy.io.fits as fits

import sys
import warnings

warnings.filterwarnings('ignore')

def imp_tbl(path):
    tbl = Table.read(path, format='ascii.csv')
    return tbl

def sersic(x,amp, r_eff, n):
    return Sersic1D.evaluate(x, amp,r_eff, n)

def exponential(x, amp, r_eff):
    return amp*np.exp(-(x/r_eff))

def mag(x, z_p):
    return -2.5*np.log10(x)+z_p

def arcsec(x):
    return (x*1.89)#
def kpc(x,d):
    return d * np.tan((np.pi/180)*((x)/3600))

def step_func(x, d):
    step = [x<d]
    return step

def sum_profile(x,i_e,r_eff,i_0,n,r_s):
    sum = sersic(x,i_e,r_eff,n) + exponential(x,i_0,r_s)#sersic(x,a2,r2,1)#
    return sum

def log_err(intens_err,intens):
    y = abs(intens_err/(intens*np.log(10)))
    return y

def redshift_d(z):
    y = (3*10**5 * z)/73
    return y*1000

path = '/volumes/ssd/test'#'article_data/ngc12'
obj_name = 'M101'

hdu = fits.getdata(path+'/'+obj_name+'.fits')
hdr = fits.getheader(path+'/'+obj_name+'.fits')

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
def radec(obj_name, catalog='simbad'):
    if catalog == 'ned':
        tbl = Ned.query_object(obj_name)
        ra,dec = tbl['RA'][0],tbl['DEC'][0]
    elif catalog == 'simbad':
        tbl = Simbad.query_object(obj_name)
        ra,dec = tbl['ra'][0],tbl['dec'][0]
    return ra,dec

wcs = WCS(hdr)
ra,dec = radec(obj_name)
sky = SkyCoord(ra=ra,dec=dec, unit='deg')
x,y = wcs.world_to_pixel(sky)

mask = fits.getdata(path+'/obj_rejec_'+obj_name+'.fits')

tbl = imp_tbl(path+'/iso_tbl_test.csv')
z = 0.02
#err, max_err, cut_idx, median_arr = iso_err_plot(path)
d = redshift_d(z)
z_p = 29
cut_i = 0
cut = 1
sma0 = tbl['sma']
pa = tbl['pa']
eps = tbl['ellipticity']

intens_r = tbl['intens']/(1.89**2)
r_err = tbl['intens_err']/(1.89**2)

#kpc = kpc(sma, d)
radius = kpc(sma0, d) #arcsec(sma) #
#sma = radius

def init_param(sma,intens):
    
    i_0 = intens[0]
    i_hr = i_0 / np.exp(1)
    abs_i = abs(intens - i_hr)
    idx_d = list(abs_i).index(np.min(abs_i))
    h = sma0[idx_d]

    r_eff = 1.86 * h
    idx_eff = list(abs(sma-r_eff)).index(np.min(abs(sma-r_eff)))
    i_e = intens[idx_eff]

    pa_eff = pa[idx_eff]
    ell_bulge = eps[idx_eff]

    pa_disk = pa[-5]
    ell_disk = eps[-5]

    return r_eff,i_e,i_0, h, pa_eff, ell_bulge, pa_disk, ell_disk

r_eff,i_e,i_0, h, pa_eff, ell_bulge, pa_disk, ell_disk = init_param(sma0, intens_r)


"""
import pyimfit
def galaxy_model(x0, y0, PA_bulge, ell_bulge, n, I_e, r_e, PA_disk, ell_disk, I_0, h):
    model = pyimfit.SimpleModelDescription()
    # define the limits on X0 and Y0 as +/-10 pixels relative to initial values
    model.x0.setValue(x0, [x0 - 10, x0 + 10])
    model.y0.setValue(y0, [y0 - 10, y0 + 10])

    bulge = pyimfit.make_imfit_function('Sersic', label='bulge')
    bulge.PA.setValue(PA_bulge, [0, 180])
    bulge.ell.setValue(ell_bulge, [0, 1])
    bulge.I_e.setValue(I_e, [0, 10*I_e])
    bulge.r_e.setValue(r_e, [0, 10*r_e])
    bulge.n.setValue(n, [0.5, 5])

    disk = pyimfit.make_imfit_function('Exponential', label='disk')
    disk.PA.setValue(PA_disk, [0, 180])
    disk.ell.setValue(ell_disk, [0, 1])
    disk.I_0.setValue(I_0, [0, 10*I_0])
    disk.h.setValue(h, [0, 10*h])

    model.addFunction(bulge)
    model.addFunction(disk)

    return model

model_desc = galaxy_model(x0=x, y0=y, PA_bulge=pa_eff, ell_bulge=ell_bulge, n=4, I_e=i_e,
                        r_e=r_eff, PA_disk=pa_disk, ell_disk=ell_disk, I_0=i_0, h=h)

imfit_fitter = pyimfit.Imfit(model_desc)
results = imfit_fitter.fit(image=hdu, mask=mask, gain=4., read_noise=1.5)
img = imfit_fitter.getModelImage()
print(results)
"""
from astropy.visualization import simple_norm

def norm(x):
    return simple_norm(x, 'linear',percent=96)

sys.exit()