import glob
import os
from astropy.io import fits

def mkdir(path, name):
    if not os.path.exists(path + '/'+name):
        os.mkdir(path +'/'+ name)

def imp(path, ext_type=0):
    if ext_type == 0:
        ext = '.fits'
    elif ext_type == 1:
        ext = '.fit'
    elif ext_type == 2:
        ext = ".csv"
    file = sorted(glob.glob(path + '/*'+ext))
    return file
    
def save_fits(path,name, data,hdr=None, ext_type=0, overwrite=True):
    if ext_type == 0:
        ext = '.fits'
    elif ext_type == 1:
        ext = '.fit'
    fits.writeto(path+'/'+name+ext, data,header=hdr, overwrite=overwrite)
    
from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
def radec(obj_name, catalog='ned'):
    if catalog == 'ned':
        tbl = Ned.query_object(obj_name)
        ra,dec = tbl['RA'][0],tbl['DEC'][0]
    elif catalog == 'simbad':
        tbl = Simbad.query_object(obj_name)
        ra,dec = tbl['ra'][0],tbl['dec'][0]
    return ra,dec

def prt_process(input):
    print(input + ' is/are done.')

#print(imp('/volumes/ssd/test/bias',1))
#mkdir('/volumes/ssd/test','test')