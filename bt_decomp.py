import numpy as np
from astropy.table import Table
from astropy.modeling.models import Sersic1D
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import sys
import warnings

warnings.filterwarnings('ignore')

def imp_tbl(path):
    tbl = Table.read(path, format='ascii.csv')
    return tbl

def sersic(x,amp, r_eff, n):
    return Sersic1D.evaluate(x, amp,r_eff, n)

def exponential(x, amp, r_s):
    return amp*np.exp(-(x/r_s))

def mag(x, z_p):
    return -2.5*np.log10(x)+z_p

def arcsec(x):
    return (x*1.89)

def kpc(x,d):
    return d * np.tan((np.pi/180)*((x)/3600))

def step_func(x, d):
    step = [x<d]
    return step

def sum_profile(x,a1,r_eff,a2,r_s):
    sum = sersic(x,a1,r_eff,4) + exponential(x,a2, r_s)
    return sum

def log_err(intens_err,intens):
    y = abs(intens_err/(intens*np.log(10)))
    return y

def redshift_d(z):
    y = (3*10**5 * z)/73
    return y*1000

from io_fits import imp, radec
from astropy.io import fits
from astropy.wcs import WCS
from ellipse_fit import detect

def init_param(x,intens,obj,hdul,mask):
    ra,dec = radec(obj)
    geo, sma = detect(hdul,mask,ra,dec,1.89)
    r_eff = sma
    i_0 = intens[0]
    i_hr = i_0 / np.exp(1)
    idx = np.where(abs(x-r_eff)==np.min(abs(x-r_eff)))
    amp_b = intens[idx]
    sma_idx = np.where(abs(intens-i_hr)==np.min(abs(intens-i_hr)))
    r_s = x[sma_idx]
    rand = np.random.randint(0,10)/10 #noise

    return amp_b[0],r_eff,i_0,r_s[0]

def decomposition(path,obj):
    hdul = fits.open(path+'/ngc4236.fits')
    mask = fits.open(path+'/obj_rejec_NGC4236.fits')[0].data

    tbl = imp_tbl(path+'/iso_tbl_test1.csv')
    intens = tbl['intens']
    intens_err = tbl['intens_err']
    z_p = 27

    sma0 = tbl['sma']
    sma = arcsec(sma0)
    popt_raw = []
    for i in range(1):
        init_list= [init_param(sma0,intens,obj,hdul,mask)]
        #print(init_list);sys.exit() #amp_b,amp_d,r_eff_1,r_eff_2,n,r_s 
        popt, pcov = curve_fit(sum_profile, sma0,intens,p0=init_list,maxfev=8000, sigma=log_err(intens_err,intens))
        popt_raw.append(popt)
    popt_arr = np.array(popt_raw)
    mena, popt,std = sigma_clipped_stats(popt_arr, axis=0, cenfunc='median',stdfunc='mad_std',sigma=3)#np.median(popt_arr,axis=0)
    #print(popt)
    a1,r1 = popt[0], popt[1]
    a2,r_s = popt[2], popt[3]

    #print(5*r_s)

    fig, ax = plt.subplots(2,1,figsize=(5,7), gridspec_kw={'height_ratios':[5,2]}, sharex=True)
    plt.subplots_adjust(hspace=0)
    
    def tick(i):
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].tick_params(axis='y', which='minor', direction='out')
        ax[i].tick_params(axis='y', which='major', direction='out')
        ax[i].tick_params(axis='x', which='minor', direction='out')
        ax[i].tick_params(axis='x', which='major', direction='out')

    ax[0].plot(sma, mag(exponential(sma,a2,r_s),z_p), label='disk', linestyle='dashed',c='C0') # a2*np.exp(-(kpc/r2)),z_p
    ax[0].plot(sma, mag(sersic(sma,a1,r1,4),z_p), label='bulge', linestyle='dashed',c='C3') # a1*np.exp(-(kpc/r1)**(1/n)),z_p
    
    ax[0].plot(sma, mag(sum_profile(sma, *popt),z_p), label='sum', linestyle='dashdot',linewidth=1,c='C2')
    #ax[0].fill_between(radius[:cut_idx], mag(max_err,z_p)-log_err(intens_err[:cut_idx], intens[:cut_idx]), mag(max_err,z_p)+err+log_err(intens_err[:cut_idx], intens[:cut_idx]), color='lightgrey') #2.5*np.log10(intens_err[:cut_idx])/2
    ax[0].scatter(sma, mag(intens, z_p),s=5,c='orange')
    ax[0].set_title(obj) #title
    ax[0].set_ylabel('$\mu_r$')
    y_bot, y_top = np.min(mag(intens,z_p))-1,np.max(mag(intens, z_p))+1
    ax[0].set_ylim(y_bot,y_top)
    ax[0].text(1,y_top-(y_top-y_bot)*0.1, 'bulge(n='+f'{4:.1f}'+') $R_{eff}=$'+f"{r1:.1f}kpc"+'\ndisk(n='+f'{1:.1f}'+') $R_{eff}=$'+f"{r_s:.1f}kpc", bbox={'boxstyle':'square', 'fc':'white'})
    ax[0].legend()
    ax[0].invert_yaxis()
    tick(0)
    """
    ax[1].errorbar(sma,mag(intens_g,22.56)-mag(intens_r,22.59), yerr=log_err(r_err,intens_r)+log_err(g_err,intens_g))
    ax[1].set_ylabel('g - r')
    """
    ax[0].axvline(x=r1, linestyle='dotted', linewidth=1.5, c='grey')
    #ax[1].axvline(x=r1, linestyle='dotted',linewidth=1.5, c='grey')
    tick(1)

    """
    ax[2].plot(kpc(sma,d), eps, '.-')
    ax[2].set_ylabel('Ellipticity')
    tick(2)
    """
    fig.supxlabel('sma(kpc)')
    plt.show()


decomposition('/volumes/ssd/test','NGC4236')

