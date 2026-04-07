from io_fits import imp, save_fits, mkdir
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
import ray

def master_bias(path,ext_type=0):
    bias_list = imp(path+'/bias', ext_type)
    master_list = []
    for i in range(len(bias_list)):
        hdu = fits.open(bias_list[i])[0].data 
        master_list.append(hdu)
    mean, master_b, std = sigma_clipped_stats(np.array(master_list,dtype=np.float32),cenfunc='median',
                                              stdfunc='mad_std',sigma=3,axis=0)
    save_fits(path+'/process','master_bias', master_b,ext_type=ext_type)
    return master_b

def master_dark(path,bias,ext_type=0):
    dark_list = imp(path+'/dark', ext_type)
    master_list = []
    for i in range(len(dark_list)):
        hdu = fits.open(dark_list[i])[0].data 
        master_list.append(hdu-bias)
    mean, master_d, std = sigma_clipped_stats(np.array(master_list,dtype=np.float32),cenfunc='median',
                                            stdfunc='mad_std',sigma=3,axis=0)
    save_fits(path+'/process','master_dark', master_d,ext_type=ext_type)
    return master_d

def db_sub(path, bias, dark, ext_type=0):
    l_list = imp(path+'/light',ext_type)
    for i in range(len(l_list)):
        n = format(i,'04')
        hdl = fits.open(l_list[i])[0]
        hdr = hdl.header
        hdu = hdl.data
        db_subed = hdu.astype(np.float32) - bias - dark
        save_fits(path+'/db_subed','db_subed'+str(n),db_subed,hdr,ext_type=ext_type)


from astropy.convolution import convolve
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

def simple_masking(arr):
    bkg_estimator = MedianBackground()
    bkg = Background2D(arr, (64,64), filter_size=(3,3), bkg_estimator=bkg_estimator)
    data = arr - bkg.background
    threshold = 1.5 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)
    seg_map = detect_sources(convolved_data, threshold, npixels=30)
    mask_map = np.array(seg_map)
    kernel = disk(10)
    mask_map_d = binary_dilation(mask_map, kernel, iterations=3)
    masked = np.where((mask_map_d!=0), 1, 0)
    return masked.astype(np.int8)

def region_mask(hdu, thrsh,pix_scale,disk_r=100,ampglow=True):
    if type(ampglow)==bool:
        if ampglow==True:
            half = disk(disk_r)
            z_arr = np.zeros_like(hdu)
            z_arr[2048-disk_r:2048,1212-disk_r-1:1212+disk_r] += half[0:disk_r,:]
    elif type(ampglow) == np.ndarray:
        half = disk(disk_r)
        z_arr0 = np.array(ampglow) #np.where(ampglow!=0,False,True)
        z_arr0[2048-disk_r:2048,1212-disk_r-1:1212+disk_r] += half[0:disk_r,:]
        z_arr = np.where(z_arr0!=0,1,0)

    else:
        half = None
        z_arr = np.where(hdu!=0, False, True)
    
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
        #aperture = EllipticalAperture(xy, 3*a, 3*b, theta)
        aperture = EllipticalAperture(xy, 3.5*a, 3.5*b, theta=theta)
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
    if type(ampglow)!=np.ndarray:
        if ampglow == True:
            masked_map[2048-disk_r:2048,1212-disk_r-1:1212+disk_r] += half[0:disk_r,:]
    elif type(half)==np.ndarray:
        if type(ampglow)==np.ndarray:
            masked_map += z_arr
    
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    return np.array(masked, dtype=np.int8)

@ray.remote
def mask(hdul,i,pix,amp_r,amp_mask=True,ext_type=0):
    hdu = fits.open(hdul[i])[0].data
    mask = region_mask(hdu, 0.99,pix,disk_r=amp_r,ampglow=amp_mask)
    n = format(i,'04')
    save_fits(path+'/mask','mask_'+str(n),data=mask,ext_type=ext_type)

    
from scipy.stats import mode

def master_flat(path, db_list,mask_loc, ext_type=0):
    scl_list = []
    m_list = []
    for i in range(len(db_list)):
        hdu = fits.open(db_list[i])[0].data
        mask = fits.open(mask_loc[i])[0].data
        tmp_flat = np.ma.masked_where(mask,hdu)
        mode0 = mode(tmp_flat[~np.isnan(tmp_flat)])[0]
        medianf = np.ma.median(tmp_flat)
        scaled = np.ma.masked_array((tmp_flat - medianf)/medianf, dtype=np.float16)
        scl_list.append(scaled)
        m_list.append(mode0.astype(np.float16))
    #m,mode1,s = sigma_clipped_stats(m_list,cenfunc='median',stdfunc='mad_std',sigma=3)
    mode1 = np.median(m_list)
    mean, sc_flat,std = sigma_clipped_stats(np.ma.masked_array(scl_list,dtype=np.float16)
                                            ,cenfunc='median',stdfunc='mad_std',sigma_lower=6,
                                            sigma_upper=3,axis=0)
    #median = np.nanmedian(sc_flat)
    #print(median);sys.exit()
    master_f = np.array(sc_flat * mode1 + mode1, dtype=np.float32)
    save_fits(path+'/process','master_flat', master_f,ext_type=ext_type)
    return master_f

def proc(db_list, flat,obj,ext_type=0):
    mean, median, std = sigma_clipped_stats(flat, cenfunc='median',
                                            stdfunc='mad_std',sigma=3)
    for i in range(len(db_list)):
        hdu = fits.open(db_list[i])[0].data
        hdr = fits.open(db_list[i])[0].header
        pp_img = hdu / (flat/median)
        n = format(i, '04')
        save_fits(path+'/pp','pp_'+obj+str(n),data=pp_img,hdr=hdr,ext_type=ext_type)

from astropy.modeling import models, fitting

def sky_model(data, bin, order=2):
        img_height, img_width = data.shape

        newImage = np.zeros((bin,bin), dtype=data.dtype)

        new_height = img_height//bin
        new_width = img_width//bin

        """
        the center position of binned pixel
        """
        xx_m = np.arange(0,img_width, img_width/bin) + new_width//2
        yy_m = np.arange(0, img_height, img_height/bin) +new_height//2

        x_m = np.array([[i for i in xx_m] for j in yy_m])
        y_m = np.array([[j for i in xx_m] for j in yy_m])

        """
        binning
        """
        for j in range(bin):
            for i in range(bin):
                y = j*new_height
                x = i*new_width
                pixel = data[y:y+new_height, x:x+new_width]
                mean, median,std = sigma_clipped_stats(pixel, cenfunc='median',stdfunc='mad_std',sigma=3)
                newImage[j,i] = median#np.nanmedian(pixel) #np.ma.median(pixel) #
                
        """
        calculate matrix x and y, these are positon component or img
        """        
        x1 = np.array([[i for i in range(img_width)] for j in range(img_height)])
        y1 = np.array([[j for i in range(img_width)] for j in range(img_height)])

        data_nc = np.ma.masked_invalid(newImage)

        """
        modeling
        """

        p_init = models.Polynomial2D(degree=order) #다항함수 모델링
        fit_p = fitting.LinearLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = fit_p(p_init, x_m, y_m, data_nc) #하늘의 모델을 반환(x,y)
        #fits.writeto(path+'/sky_model.fits',model(x1,y1),overwrite=True)
        return model(x1, y1)

from rbf_skysub import rbf_sky_model
@ray.remote
def sky_sub(pp_list, mask_list,obj,i,ext_type=0):
        n = format(i,'04')
        hdu = fits.open(pp_list[i])[0].data 
        hdr = fits.open(pp_list[i])[0].header
        mask = fits.open(mask_list[i])[0].data
        m_data = np.ma.masked_array(hdu, mask, dtype=np.float32)#np.where(mask!=0,np.nan, hdu) #np.ma.masked_where(mask, hdu) #
        sky = sky_model(m_data, 64).astype(np.float32)
        subed = np.array(hdu-sky).astype(np.float32)
        hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        save_fits(path+'/sky_subed',obj+'_'+str(n),data=subed,hdr=hdr,ext_type=ext_type)
      

from io_fits import radec, prt_process  
def astrometry(path, obj_name, radius,ext_type=0):
    if ext_type == 0:
        ext = '.fits'
    else :
        ext = '.fit'
    ra,dec = radec(obj_name)
    file = open(path+'/'+obj_name+'.sh', 'w')
    file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots *'+ext+' \nrm -rf *.xyls *.axy *.corr *.match *.new *.rdls *.solved\nulimit -n 4096')
    file.close()


#img_list = imp(path, ext_type=0) #dtype=list

def full_proc(path,obj,ext_type):
    """
    mkdir(path, 'process')
    bias = master_bias(path,ext_type)
    dark = master_dark(path, bias,ext_type)
    
    amp_mask = simple_masking(dark)
    
    mkdir(path,'db_subed')
    db_sub(path, bias, dark,ext_type=ext_type)
    prt_process('db_subd')
    
    hdul = imp(path+'/db_subed',ext_type=ext_type)
    
    mkdir(path, 'mask')
    ray.init(num_cpus=8)
    ray.get([mask.remote(hdul,i,pix=1.89,amp_r=300,
                         amp_mask=amp_mask,ext_type=ext_type) for i in range(len(hdul))])
    ray.shutdown()
    prt_process('masking')
    """
    mask_list = imp(path+'/mask',ext_type=ext_type)
    """
    flat = master_flat(path, hdul, mask_list,ext_type=ext_type)
    prt_process('making flat')
    #flat = fits.open(path+'/process/master_flat.fit')[0].data
    mkdir(path,'pp')
    proc(hdul,flat,obj,ext_type=ext_type)
    prt_process('pp')
    """
    pp_list = imp(path+'/pp',ext_type=ext_type)
    mkdir(path,'sky_subed')
    
    """
    for i in range(len(pp_list)):
        sky_sub(pp_list,mask_list,obj,i,ext_type=ext_type)
    """
    
    ray.init(num_cpus=8)
    ray.get([sky_sub.remote(pp_list,mask_list,obj,i,ext_type=ext_type) for i in range(len(pp_list))])
    ray.shutdown()
    prt_process('sky sub')
    """
    astrometry(path,obj,1.5,ext_type=1)
    prt_process('proc')
    sys.exit()
    """
path = '/volumes/ssd/u_test'    
full_proc(path,'NGC784',1)
"""
obj = 'Arp142'
hdu = fits.open(path+'/pp/pp_'+obj+'0000.fit')[0].data 
hdu_mask = fits.open(path+'/mask/mask_0000.fit')[0].data 
data = np.ma.masked_array(hdu, hdu_mask)
sky_model(data, 64)
"""