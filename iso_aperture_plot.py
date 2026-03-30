import numpy as np
import astropy.io.fits as fits
from photutils.aperture import EllipticalAperture
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import sys

path = '/volumes/ssd/test'
#color = 'r'
title = 'NGC 4236'
hdl = fits.open(path+'/ngc4236.fits')
model = fits.open(path+'/model.fits')[0].data
hdu = hdl[0].data
hdr = hdl[0].header
wcs = WCS(hdr)
tbl = Table.read(path+'/iso_tbl_test.csv', format='ascii.csv')
mean, median, std = sigma_clipped_stats(hdu, cenfunc='median', stdfunc='mad_std', sigma=3)

sma = tbl['sma'][1:]
x0, y0 = tbl['x0'][1:], tbl['y0'][1:]
eps = tbl['ellipticity'][1:]
pa = tbl['pa'][1:]
#print(pa*np.pi/180)

#sys.exit()
median = np.median(hdu)#[1500-30:1500+300,1500-300:1500+300])
std = np.std(hdu)#[1500-300:1500+300,1500-300:1500+300])
fig, ax = plt.subplots(1,3, figsize=(15,5), subplot_kw=dict(projection=wcs), sharex=True, sharey=True) #plt.subplots(subplot_kw=dict(projection=wcs))#
fig.subplots_adjust(wspace=0.3)
ax[0].imshow(hdu,vmax=median+3*std, vmin=median-3*std,origin='lower')# median-9*std ,[1500-350:1500+350,1500-350:1500+350],
ax[1].imshow(model,vmax=median+3*std, vmin=median-3*std,origin='lower')
ax[2].imshow(hdu-model,vmax=median+3*std, vmin=median-3*std,origin='lower')# median-9*std

for i in range(len(sma)):
    if i%1 == 0:
        n = i
        aper = EllipticalAperture(positions=(x0[n],y0[n]), a=sma[n], b=sma[n]*(1-eps[n]), theta=pa[n]*np.pi/180)
        aper.plot(ax=ax[0],color='white', linewidth=0.7)

#ax[0].set_xlabel('R.A.')
ax[0].coords[0].set_auto_axislabel(False)
ax[0].coords[1].set_auto_axislabel(False)
#ax[0].set_ylabel('DEC')
ax[0].set_title('Raw Image')
ax[1].coords[0].set_auto_axislabel(False)
#ax[1].set_xlabel('R.A.')
ax[1].coords[1].set_auto_axislabel(False)
#ax[1].set_ylabel('DEC')
ax[1].set_title('Model')
#ax[2].set_xlabel('R.A.')
ax[2].coords[0].set_auto_axislabel(False)
ax[2].coords[1].set_auto_axislabel(False)
#ax[2].set_ylabel('DEC')
ax[2].set_title('Model-Image')
"""
ax.coords[0].set_auto_axislabel(False)
ax.coords[1].set_auto_axislabel(False)
"""
fig.suptitle(title)
fig.supylabel('DEC')
fig.supxlabel('R.A.')

plt.show()