import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources, SegmentationImage
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
from photutils.detection import find_peaks
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import sys

def obj_rej_mask(hdu, thrsh,hdr,ra,dec):
    mask1 = np.where(hdu!=0, False, True)
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est, mask=mask1)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=9, mask=mask1) #1차 천체 탐지
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000, nlevels=32, contrast=0.0005,
                               progress_bar=False) #천체분리

    segm_d = np.array(segm_deblend).astype(np.int32)

    cat = SourceCatalog(hdu,segm_deblend, convolved_data=conv_hdu)

    a_list = list(cat.semiminor_sigma.value)
    
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
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
    cat1 = cat[idx]
 
    segm_d[cat1.bbox_ymin[0]:cat1.bbox_ymax[0],cat1.bbox_xmin[0]:cat1.bbox_xmax[0]] = 0
    #plt.imshow(segm_d,origin='lower');plt.show();sys.exit()
    segm = SegmentationImage(segm_d)
    cat = SourceCatalog(segm, segm, convolved_data=conv_hdu)
     
    a_list = list(cat.semiminor_sigma.value)
    
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-3:]
    top_idx = [a_list.index(x) for x in tmp_num]
    for i in top_idx:
        """
        g_aper = l[i]
        a = g_aper.a
        b = g_aper.b
        xypos = g_aper.positions
        theta = g_aper.theta
        xy = (int(xypos[0]), int(xypos[1]))
        """
        cat0 = cat[i]
        xy = (cat0.xcentroid, cat0.ycentroid)
        theta = cat0.orientation.value *np.pi /180
        a,b = 3*cat0.semimajor_sigma.value, 3*cat0.semiminor_sigma.value
        aperture = EllipticalAperture(xy, 2.5*a, 2.5*b, theta)
        mask = np.array(aperture.to_mask(method='center')).astype(np.int8)
        mask_x, mask_y = mask.shape
    
        st_x = np.int16(xy[1] - mask_x/2)
        st_y = np.int16(xy[0] - mask_y/2)
    
        x, y = hdu.shape
   
        def lim(st, mask_s, arr_s):
            if st < 0 and st+mask_s<arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s
            elif st<0 and st+mask_s>arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s + st - arr_s
        
            elif st+mask_s > arr_s:
                arr_st = st
                mask_st = 0
                mask_l = arr_s - st

            else:
                arr_st = st
                mask_st = 0
                mask_l = mask_s
            return arr_st, mask_st, mask_l
        
        arr_x, mask_s_x, mask_l_x = lim(st_x, mask_x, x)
        arr_y, mask_s_y, mask_l_y = lim(st_y, mask_y, y)
        mask = mask[mask_s_x:mask_l_x,mask_s_y:mask_l_y] 
        m_x, m_y = mask.shape #crop mask
        arr_zero[arr_x:arr_x+m_x, arr_y:arr_y+m_y] += mask
    kernel0 = disk(3) 
    segm_d = binary_dilation(segm_d, kernel0, iterations=1) #ngrow
    masked_map = np.where(segm_d!=0, 1, 0) + arr_zero #region 마스크 영상과 segmentation 마스크 영상을 합침
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    
    return np.array(masked, dtype=np.int8)

import warnings


warnings.filterwarnings('ignore')
hdl = fits.open('/volumes/ssd/test/ngc4236.fits')[0]
hdu = hdl.data 
hdr = hdl.header

#mask = fits.open('/volumes/ssd/intern/25_summer/M101_L/mask_coadd.fits')[0].data
#x,y = hdu.shape
from io_fits import radec
ra,dec = radec('NGC4236')
mask = obj_rej_mask(hdu,3.,hdr,ra,dec)
#plt.imshow(mask, origin='lower')
map = np.where(mask!=0, np.nan, hdu)
#fits.writeto('/volumes/ssd/test/obj_rejec_NGC4236.fits', mask, overwrite=True)
#map1 = np.where(map==0,np.nan, map)
#plt.imshow(mask, origin='lower')
#plt.imshow(hdu, origin='lower') #vmax=median+3*std, vmin=median-3*std,
median = np.nanmedian(map)
std = np.nanstd(map)
plt.imshow(map,vmax=median+3*std, vmin=median-3*std ,origin='lower')
#plt.imshow(map[int(x/2-1300):int(x/2+1300),int(y/2-1300):int(y/2+1300)],vmax=median+3*std, vmin=median-3*std,
            #origin='lower')
#plt.colorbar()
plt.show()
