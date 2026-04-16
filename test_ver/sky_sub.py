from astropy.modeling import models, fitting
import numpy as np
import warnings
from astropy.io import fits
from io_fits import imp, save_fits
import ray
import matplotlib.pyplot as plt
import sys
from photutils.segmentation import make_2dgaussian_kernel
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats

def seg_sky_model(data, bin):
        img_height, img_width = data.shape

        #newImage = np.zeros((bin,bin), dtype=data.dtype)

        new_height = img_height//bin
        new_width = img_width//bin

        """
        the center position of binned pixel
        """
        xx_m = np.arange(0,img_width, img_width/bin)
        yy_m = np.arange(0, img_height, img_height/bin)

        x_m = np.array([[i for i in xx_m] for j in yy_m])
        y_m = np.array([[j for i in xx_m] for j in yy_m])

        """
        binning
        """
        bin_img = np.zeros((data.shape[0],data.shape[1]))
        for j in range(bin):
            for i in range(bin):
                y = j*new_height
                x = i*new_width
                pixel = np.ma.masked_invalid(data[y:y+new_height, x:x+new_width])
                #plt.imshow(pixel,origin='lower');plt.show();sys.exit()
                y_arr, x_arr = np.mgrid[:np.shape(pixel)[0],:np.shape(pixel)[1]]
                
                p_init = models.Polynomial2D(degree=2) #다항함수 모델링
                fit_p = fitting.LinearLSQFitter()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = fit_p(p_init, x_arr, y_arr, pixel)
                bin_img[y:y+new_height,x:x+new_width] = model(x_arr,y_arr) #np.ma.median(pixel) #
        
        #kernel = make_2dgaussian_kernel(5/1.86,15)
        #plt.imshow(kernel);plt.show();sys.exit()
        #model_f = convolve(bin_img,kernel)
        """
        from scipy.interpolate import NearestNDInterpolator,RegularGridInterpolator
        x, y = np.arange(0,data.shape[0], bin), np.arange(0,data.shape[1], bin)
        x_mesh,y_mesh = np.meshgrid(x,y, indexing='ij')
        interp = RegularGridInterpolator((x,y), bin_img, method='linear')
        bkg_intrp = interp(x_mesh, y_mesh)
        return bkg_intrp
        """
        #fits.writeto('/volumes/ssd/Arp142/seg_sky_model.fits',bin_img, overwrite=True);sys.exit()
        return bin_img

path = ""
@ray.remote
def sky_sub(pp_list, mask_list,obj,i,ext_type=0):
        n = format(i,'04')
        hdu = fits.open(pp_list[i])[0].data 
        hdr = fits.open(pp_list[i])[0].header
        mask = fits.open(mask_list[i])[0].data
        m_data = np.where(mask!=0,np.nan, hdu) #np.ma.masked_where(mask, hdu) #
        sky = seg_sky_model(m_data, 64).astype(np.float32)
        subed = np.array(hdu-sky).astype(np.float32)
        hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        save_fits(path+'/sky_subed',obj+'_'+str(n),data=subed,hdr=hdr,ext_type=ext_type)

hdu = fits.open('/volumes/ssd/Arp142/pp/pp_Arp1420000.fit')[0].data 
mask = fits.open('/volumes/ssd/Arp142/mask/mask_0000.fit')[0].data 
img = np.ma.masked_array(hdu, mask=mask)

mena, median, std = sigma_clipped_stats(img, cenfunc='median',stdfunc='mad_std',sigma=3)

seg_model = seg_sky_model(img, 8)

median_seg, std_seg = np.median(seg_model), np.std(seg_model)
#fits.open('/volumes/ssd/test/conv_seg_sky_model.fits')[0].data 
normal_model = fits.open('/volumes/ssd/Arp142/sky_model.fits')[0].data 
residual = normal_model - seg_model
resi_median, resi_std = np.median(residual),np.std(residual)
fig, ax = plt.subplots(1,3,sharex=True, sharey=True)
ax[0].imshow(normal_model,origin='lower')
ax[0].set_title('normal model')
ax[1].imshow(seg_model, origin='lower')#, vmax=median_seg+3*std_seg, vmin=median_seg-3*std_seg)
ax[1].set_title('segment sky model')
ax[2].imshow(normal_model-seg_model,origin='lower')#,vmax=resi_median+3*resi_std,vmin=resi_median-3*resi_std)
ax[2].set_title('normal - segmented')
#fig.colorbar(resi_cmap)
plt.show()
