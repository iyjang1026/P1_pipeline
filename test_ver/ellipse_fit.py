import numpy as np
import sys
import astropy.io.fits as fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve
import astropy.units as u
from photutils.segmentation import SegmentationImage, detect_sources, deblend_sources ,make_2dgaussian_kernel, SourceCatalog
from photutils.background import MedianBackground, Background2D
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model, IsophoteList, Isophote
from photutils.aperture import EllipticalAperture
import matplotlib.pyplot as plt
import os


def detect(hdul, mask, ra,dec,pix,i=0):
    hdu0 = hdul[0].data
    hdr = hdul[0].header
    hdu = np.ma.masked_where(mask,np.ma.masked_equal(hdu0, np.zeros(shape=hdu0.shape))-i) #테두리와 region 마스크를 적용
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est) #배경추출
    data = hdu - bkg.background #background 제거
    threshold = 3.*bkg.background_rms #threshold 설정
    kernel = make_2dgaussian_kernel(fwhm=3.0/pix, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5) #1차 천체 탐지
    
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000, nlevels=32, contrast=0.0005,
                               progress_bar=False) #중심부 M 101 탐지
    #plt.imshow(segm_deblend, origin='lower'); plt.show(); sys.exit()
    cat = SourceCatalog(hdu, segm_deblend, convolved_data=conv_hdu)
    a_list = list(cat.semiminor_sigma.value)
     
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
    top_idx = [a_list.index(x) for x in tmp_num]
    cat = cat[top_idx]
    x,y = cat.xcentroid,cat.ycentroid
    w = WCS(hdr)
    eq_coord = SkyCoord(ra,dec,frame='fk5',unit='deg')
    pix_x,pix_y = w.world_to_pixel(eq_coord)
    idx = np.where((np.min(abs(pix_x-x))==abs(pix_x-x))&(np.min(abs(pix_y-y)==abs(pix_y-y))))
    obj = cat[idx][0]

    aper = EllipticalAperture((obj.xcentroid, obj.ycentroid), obj.semimajor_sigma.value*3, obj.semiminor_sigma.value*3, obj.orientation.value*np.pi/180)
    x,y = obj.xcentroid, obj.ycentroid #obj.positions
    geometry = EllipseGeometry(x0=x, y0=y, sma=obj.semimajor_sigma.value*3, eps=obj.ellipticity.value, pa=obj.orientation.value*np.pi/180) #isophote를 위한 초기 타원 생성
    #plt.imshow(hdu, origin='lower'); aper.plot(color='C3'); plt.show(); sys.exit()
    return geometry, obj.semimajor_sigma.value*3
    #hdu1 = np.ma.masked_where(hdu>40000, hdu)
    
from io_fits import norm    
def ellipse(path,hdul,mask, geometry, sma):
    hdu = hdul.data
    hdr = hdul.header
    #norm_hdu = (hdu - np.min(hdu))/(np.max(hdu)-np.min(hdu))    
    hdu0 = np.ma.array(hdu,mask=mask)
    ellipse = Ellipse(hdu0, geometry)
    isolist = ellipse.fit_image(sma0=0.8*sma,minsma=0.01*sma, maxsma=3*sma,integrmode='median',step=0.1, sclip=3.0, nclip=3, fflag=0.3, fix_center=True) #isophote
    tbl = isolist.to_table()    
    tbl.write(path+'/iso_tbl_NGC4236.csv', format='ascii.csv', overwrite=True)
    #fits.writeto(path+'/norm_coadd.fits', norm_hdu, header=hdr, overwrite=True)
    print(tbl)
    #fill = np.median(hdu[1050:2000,1200:2000])
    
    model = build_ellipse_model(hdu0.shape, isolist) #modeling
    return model, tbl#, np.ma.std(hdu)

#from io_fits import radec
"""
path = '/volumes/ssd/test/'
hdul = fits.open(path+'/NGC4236.fits')
mask = fits.open(path+'/obj_rejec_NGC4236.fits')[0].data 
ra,dec = radec('NGC4236')
geo,sma = detect(hdul,mask,ra,dec,1.89)
model, tbl = ellipse(path,hdul[0],mask,geo,sma)
fits.writeto(path+'/model_NGC4236.fits',model,overwrite=True)
sys.exit()
"""