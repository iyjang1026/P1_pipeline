import numpy as np
from utils import file_list, mkdir, prt_process
from frameproc import Master, Process
from masking import simple_masking
import ray
import sys

@ray.remote
def mask(path=str,hdul=list,i=int,pix=float,amp_r=bool|np.ndarray,amp_mask=True,ext_type=0):
    Process.mask(path, hdul, i, pix, amp_r, amp_mask,ext_type)

def full_proc(path,obj,ext_type):
    """
    mkdir(path, 'process')
    bias = Master.master_bias(path,ext_type)
    dark = Master.master_dark(path, bias,ext_type)
    
    amp_mask = simple_masking(dark)
    
    mkdir(path,'db_subed')
    Process.db_sub(path, bias, dark,ext_type=ext_type)
    prt_process('db_subd')
    """
    hdul = file_list(path+'/db_subed',ext_type=ext_type)
    """
    mkdir(path, 'mask')
    ray.init(num_cpus=8)
    ray.get([mask.remote(path,hdul,i,pix=1.89,amp_r=300,
                         amp_mask=amp_mask,ext_type=ext_type) for i in range(len(hdul))])
    ray.shutdown()
    prt_process('masking')
    """
    mask_list = file_list(path+'/mask',ext_type=ext_type)
    """
    flat = Master.master_flat(path, hdul, mask_list,ext_type=ext_type)
    prt_process('making flat')
    #flat = fits.open(path+'/process/master_flat.fit')[0].data
    mkdir(path,'pp')
    Process.proc(path,hdul,flat,obj,ext_type=ext_type)
    prt_process('pp')
    """
    pp_list = file_list(path+'/pp',ext_type=ext_type)
    mkdir(path,'sky_subed')
    #@ray.remote
    def sky_sub(path=str,pp_list=list, mask_list=list,obj=str,i=int,model='polynomial',bin=64,ext_type=0):
        Process.sky_sub(path, pp_list, mask_list, obj,i,model, bin, ext_type)
    
    for i in range(len(pp_list)):
        sky_sub(path,pp_list,mask_list,obj,i,ext_type=ext_type)
    """
    #parallel processing if your computing power is enough to process 
    ray.init(num_cpus=8)
    ray.get([sky_sub.remote(path,pp_list,mask_list,obj,ext_type=ext_type) for i in range(len(pp_list))])
    ray.shutdown()
    """
    prt_process('sky sub')
    
    Process.astrometry(path,obj,1.5,ext_type=1)
    prt_process('proc')
    sys.exit()

path = '/volumes/ssd/u_test'    
full_proc(path,'NGC784',1)