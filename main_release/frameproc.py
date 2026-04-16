from utils import file_list, save_fits
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

from sky import poly_sky_model, rbf_sky_model
from masking import region_mask
from utils import radec 

class Master:
    def master_bias(path=str,ext_type=0):
        bias_list = file_list(path+'/bias', ext_type)
        master_list = []
        for i in range(len(bias_list)):
            hdu = fits.open(bias_list[i])[0].data 
            master_list.append(hdu)
        mean, master_b, std = sigma_clipped_stats(np.array(master_list,dtype=np.float32),cenfunc='median',
                                                stdfunc='mad_std',sigma=3,axis=0)
        save_fits(path+'/process','master_bias', master_b,ext_type=ext_type)
        return master_b

    def master_dark(path=str,bias=np.ndarray,ext_type=0):
        dark_list = file_list(path+'/dark', ext_type)
        master_list = []
        for i in range(len(dark_list)):
            hdu = fits.open(dark_list[i])[0].data 
            master_list.append(hdu-bias)
        mean, master_d, std = sigma_clipped_stats(np.array(master_list,dtype=np.float32),cenfunc='median',
                                                stdfunc='mad_std',sigma=3,axis=0)
        save_fits(path+'/process','master_dark', master_d,ext_type=ext_type)
        return master_d

    def master_flat(path=str, db_list=list,mask_list=list, ext_type=0):
        scl_list = []
        m_list = []
        for i in range(len(db_list)):
            hdu = fits.open(db_list[i])[0].data
            mask = fits.open(mask_list[i])[0].data
            tmp_flat = np.ma.masked_where(mask,hdu)
            medianf = np.ma.median(tmp_flat)
            scaled = np.ma.masked_array((tmp_flat - medianf)/medianf, dtype=np.float16)
            scl_list.append(scaled)
            m_list.append(medianf.astype(np.float16))
        median1 = np.median(m_list)
        mean, sc_flat,std = sigma_clipped_stats(np.ma.masked_array(scl_list,dtype=np.float16)
                                                ,cenfunc='median',stdfunc='mad_std',sigma_lower=6,
                                                sigma_upper=3,axis=0)
        master_f = np.array(sc_flat * median1 + median1, dtype=np.float32)
        save_fits(path+'/process','master_flat', master_f,ext_type=ext_type)
        return master_f

class Process:
    def db_sub(path=str, bias=np.ndarray, dark=np.ndarray, ext_type=0):
        l_list = file_list(path+'/light',ext_type)
        for i in range(len(l_list)):
            n = format(i,'04')
            hdl = fits.open(l_list[i])[0]
            hdr = hdl.header
            hdu = hdl.data
            db_subed = hdu.astype(np.float32) - bias - dark
            save_fits(path+'/db_subed','db_subed'+str(n),db_subed,hdr,ext_type=ext_type)

    def mask(path=str,hdul=list,i=int,pix=float,amp_r=bool|np.ndarray,amp_mask=True,ext_type=0):
        hdu = fits.open(hdul[i])[0].data
        mask = region_mask(hdu, 0.99,pix,disk_r=amp_r,ampglow=amp_mask)
        n = format(i,'04')
        save_fits(path+'/mask','mask_'+str(n),data=mask,ext_type=ext_type)

    def proc(path, db_list, flat,obj,ext_type=0):
        mean, median, std = sigma_clipped_stats(flat, cenfunc='median',
                                                stdfunc='mad_std',sigma=3)
        for i in range(len(db_list)):
            hdu = fits.open(db_list[i])[0].data
            hdr = fits.open(db_list[i])[0].header
            pp_img = hdu / (flat/median)
            n = format(i, '04')
            save_fits(path+'/pp','pp_'+obj+str(n),data=pp_img,hdr=hdr,ext_type=ext_type)

    def sky_sub(path=str,pp_list=list, mask_list=list,obj=str,i=int,model='polynoimal',bin=64,ext_type=0):
        n = format(i,'04')
        hdu = fits.open(pp_list[i])[0].data 
        hdr = fits.open(pp_list[i])[0].header
        mask = fits.open(mask_list[i])[0].data
        m_data = np.ma.masked_array(hdu, mask, dtype=np.float32)
        if model == 'polynomial':
            sky = poly_sky_model(m_data,bin).astype(np.float32)
        elif model == 'rbf':
            if bin > 16:
                raise ValueError("bin must be smaller than 16 at rbf modeling")
            else:
                sky = rbf_sky_model(m_data, bin).astype(np.float32)
        
        subed = np.array(hdu-sky).astype(np.float32)
        hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        save_fits(path+'/sky_subed',obj+'_'+str(n),data=subed,hdr=hdr,ext_type=ext_type)
        
    def astrometry(path=str, obj_name=str, radius=float,ext_type=0):
        if ext_type == 0:
            ext = '.fits'
        else :
            ext = '.fit'
        ra,dec = radec(obj_name)
        file = open(path+'/'+obj_name+'.sh', 'w')
        file.write(f'solve-field --index-dir /Users/jang-in-yeong/solve/index4100 --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots *'+ext+' \nrm -rf *.xyls *.axy *.corr *.match *.new *.rdls *.solved; ulimit -n 4096')
        file.close()
