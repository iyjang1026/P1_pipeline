import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats

from astropy.convolution import convolve
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

import warnings
warnings.filterwarnings('ignore')

def region_mask(hdu, thrsh,pix_scale,ampglow=True):
    if ampglow == True:
        half = disk(100)
        z_arr = np.zeros_like(hdu)
        z_arr[2048-100:2048,1212-100-1:1212+100] += half[0:100,:]
    else :
        half = None
        z_arr = np.where(hdu!=0,False,True)
    
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(5,5), bkg_estimator=bkg_est, mask=z_arr)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3/pix_scale, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5, mask=z_arr)
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000,connectivity=8, mode='exponential', nlevels=32, contrast=0.001,
                               progress_bar=False)
    seg = np.array(seg_map)
    
    cat = SourceCatalog(data, segm_deblend, convolved_data=conv_hdu)

    a_list = list(cat.semiminor_sigma.value)
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
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
        aperture = EllipticalAperture(xy, 3*a, 3*b, theta)
        #aperture = #EllipticalAperture(xy, 3.5*a, 3.5*b, theta=theta)
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
    seg_d= binary_dilation(seg, kernel0, iterations=1)
    masked_map = np.where(seg_d!=0, 1, 0) + arr_zero
    if half != None:
        masked_map[2048-100:2048,1212-100-1:1212+100] += half[0:100,:]
    
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    
    return np.array(masked, dtype=np.int8)

def bkg_std(hdu, mask, size):
    std_list = []
    median_list = []
    arr = np.ma.masked_where(mask, np.ma.masked_equal(hdu, 0))#np.where(mask!=0, np.nan, hdu) #
    """
    mean, median, std = sigma_clipped_stats(arr, cenfunc='median', stdfunc='mad_std', sigma=3.)
    return std
    """
    #plt.imshow(arr1, origin='lower'); plt.show(); sys.exit()
    x,y = arr.shape
    center_x, center_y = int(x/2), int(y/2)

    for i in range(1000):
        rand_st_x, rand_st_y = np.random.randint(center_x-1500, center_x+1500-size, 2)
        bin_arr = arr[rand_st_x:rand_st_x+size, rand_st_y:rand_st_y+size]
        mean, median1, std1 = sigma_clipped_stats(bin_arr, cenfunc='median', stdfunc='mad_std', sigma=3)
        #print(median1, std1);sys.exit()
        median_list.append(median1)
        std_list.append(std1)
    mean, std_median, std = sigma_clipped_stats(np.array(std_list).astype(np.float32),
                                            cenfunc='median', stdfunc='mad_std', sigma=3.)
    print(std_median)
    return std_median, np.array(median_list, dtype=np.float32)

def sb_limit(path,obj,pix,std_noise,color):
    #read catalogue
    source = path + '/sky_subed/coadd.cat'#
    data = Table.read(source, format='ascii', converters={'obsid':str})
    #check the catalogue location
    sdss = Table.read(path + '/sdss_'+obj+'.csv', format='ascii') #check!! 

    #extract coordinate
    sdsscat = sdss['ra', 'dec', 'g','r']
    objcat = data['ALPHAPEAK_J2000','DELTAPEAK_J2000','FLUX_BEST', 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE']
    #obj_cat = objcat[(objcat['ERRAWIN_IMAGE']<0.01)&(objcat['ERRBWIN_IMAGE']<0.01)]
    sdss_coord = SkyCoord(ra=sdsscat['ra']*u.degree, dec=sdsscat['dec']*u.degree, frame='fk5')
    obj_coord = SkyCoord(ra=objcat['ALPHAPEAK_J2000'], dec=objcat['DELTAPEAK_J2000'], frame='fk5')


    idx1, d2d1, d3d1 = sdss_coord.match_to_catalog_sky(obj_coord)
    sdss_data = sdsscat
    obj_f = objcat[idx1]


    obj_flux = obj_f['FLUX_BEST']
    sdss_mag = sdss_data[color]
    count = np.array(obj_flux)
    mag = np.array(sdss_mag)
    g = sdss_data['g']
    r = sdss_data['r']

    from astropy.stats import sigma_clipped_stats

    m = -2.5*np.log10(count)


    mM = mag - m
    #plt.scatter(mag, mM);plt.show();sys.exit()


    from scipy.optimize import curve_fit

    def log(x,a,c):
        return a*np.log10(x)+c 
    
    mean, median1, std = sigma_clipped_stats(mM, cenfunc='median', stdfunc='mad_std', sigma=3.0)
    z = np.where((mM<=median1+3*std)&(mM>=median1-3*std), mM, np.nan) # sigma_clip(mM, cenfunc='median', stdfunc='mad_std', sigma=3) #
    #mag_ob = m + z
    r1 = r[~np.isnan(z)]
    g1 = g[~np.isnan(z)]
    def std_formular(count1,a,z1):
        return -2.5*np.log10(count1) + a*(g1-r1) + z1
    
    t_r = count[~np.isnan(z)]
    if color == 'r':
        c = r1
    else :
        c = g1
    popt,pcov = curve_fit(std_formular,t_r,c)
    print(popt)
    #print(np.median(popt[0]*(g1-r1)+popt[1]))
    def line(x,a,b):
        return a*(-2.5*np.log10(x)) +b
    
    popt_line,pcov_line = curve_fit(line,t_r,std_formular(t_r,*popt))
    #print(popt_line[0]*(-2.5))

    plt.scatter(count,mag,s=2,c='grey')
    plt.scatter(t_r,std_formular(t_r,*popt),s=2,c='r')

    count.sort()
    plt.plot(count,line(count,*popt_line),c='k',linewidth=1.5)

    plt.xscale('log', base=10)
    plt.xlabel('Flux(log10)')
    plt.ylabel('$\mu_{SDSS,r}$')
    sb_lim = popt[1] - 2.5*np.log10(std_noise/(pix*10))
    plt.text(10**3.5, 10, f'$Z_p$ = {popt[1]:.2f}\nSB Limit = {sb_lim:.2f}', bbox={'boxstyle':'square', 'fc':'white'})
    plt.title(f'{color}-band SB limit of {obj}')
    print(f'Z_p is {popt[1]}')
    print(f'SB Limit is {sb_lim}')
    plt.show()

#from ccdproc import region_mask

def sb_limit_proc(path,obj,pix,color):
    hdu = fits.open(path+'/sky_subed/coadd.fits')[0].data
    mask = region_mask(hdu,1.,1.89,ampglow=False)
    #mask = fits.open(path+'/mask_IC3280_r.fits')[0].data#region_mask(hdu, 0.99,pix,ampglow=False)
    std_noise, median_arr = bkg_std(hdu,mask,128)
    sb_limit(path,obj,pix,std_noise,color)


sb_limit_proc('/volumes/ssd/Arp142','Arp142',1.89,'r')