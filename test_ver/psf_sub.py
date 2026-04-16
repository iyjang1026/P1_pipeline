from astropy.io import fits
from photutils.psf import CircularGaussianPRF, make_psf_model_image
from photutils.detection import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import sys

hdl = fits.open('/volumes/ssd/Arp142/sky_subed/coadd.fits')[0]
hdu = hdl.data 
hdr = hdl.header 
mask = fits.getdata('/volumes/ssd/Arp142/psf_obj_mask.fits')

from astropy.wcs import WCS

wcs = WCS(hdr)
raw = np.where(mask!=0, np.nan, hdu)#np.ma.masked_array(hdu, mask)
x_cen,y_cen = raw.shape
crop = 512
img = hdu[(x_cen-crop)//2:(x_cen+crop)//2,(y_cen-crop)//2:(y_cen+crop)//2]
data = raw[(x_cen-crop)//2:(x_cen+crop)//2,(y_cen-crop)//2:(y_cen+crop)//2]
#plt.imshow(np.log10(data), origin='lower');plt.show();sys.exit()
#data = np.ma.masked_equal(hdu,0)
peaks_tbl = find_peaks(data,threshold=50000,wcs=wcs)

from astropy.nddata import NDData, StdDevUncertainty
from photutils.background import Background2D, MedianBackground
bkg_est = MedianBackground()
bkg = Background2D(data,(64,64),filter_size=(5,5),bkg_estimator=bkg_est)
error = bkg.background_rms
uncertainty = StdDevUncertainty(error)
nddata = NDData(data=data, uncertainty=uncertainty)

size = 25
hsize = (size - 1) / 2
x = peaks_tbl['x_peak']
y = peaks_tbl['y_peak']
mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
        (y > hsize) & (y < (data.shape[0] -1 - hsize)))

from astropy.table import Table
stars_tbl = Table()
stars_tbl['x'] = x[mask]
stars_tbl['y'] = y[mask]

from photutils.psf import extract_stars
stars = extract_stars(nddata, stars_tbl, size=25)

from photutils.segmentation import make_2dgaussian_kernel
kernel = np.array(make_2dgaussian_kernel(5/1.89, size=25))

from photutils.psf import EPSFBuilder
epsf_builder = EPSFBuilder(oversampling=3, maxiters=3,
                           progress_bar=False,smoothing_kernel=kernel)
epsf, fitted_stars = epsf_builder(stars)

from photutils.detection import DAOStarFinder, StarFinder
from photutils.psf import PSFPhotometry
psf_model = epsf
fit_shape = (7,7)
finder = DAOStarFinder(6., 3)
#tbl = finder.find_stars(data)
#print(tbl);sys.exit()
psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder, aperture_radius=7)

phot = psfphot(nddata, error=error)
model = psfphot.make_model_image(np.shape(data))
residual = psfphot.make_residual_image(img)
fits.writeto('/volumes/ssd/Arp142/psf_sub_coadd.fits',residual, overwrite=True)
fits.writeto('/volumes/ssd/Arp142/psf.fits',psf_model.data, overwrite=True)

from astropy.visualization import simple_norm
def norm(x):
    return simple_norm(x, 'log', percent=99.0)

fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
ax[0].imshow(data,norm=norm(data),origin='lower')
ax[1].imshow(model, norm=norm(model), origin='lower')
ax[2].imshow(residual, norm=norm(residual), origin='lower')
plt.show()

