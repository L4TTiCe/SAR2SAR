from PIL import Image, UnidentifiedImageError
from pathlib import Path
from glob import glob
import numpy as np
import os
import argparse
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = argparse.ArgumentParser(description='Script to compute speckle metrics for image pairs')
parser.add_argument('--noisy-dir', dest='noisy_dir', help='Path to the noisy(with speckles) images directory to be considered')
parser.add_argument('--filtered-dir', dest='filtered_dir', help='Path to the filtered(without speckles) images directory to be considered')
parser.add_argument('--img-ext', dest='ext', help='Extension of the Images to be considered')
# parser.add_argument('--noisy', dest='noisy', help='Path to the noisy(with speckles) image to be considered')
# parser.add_argument('--filtered', dest='filtered', help='Path to the filtered(without speckles) image to be considered')
args = parser.parse_args()

def SSI(noisy, filtered):
    """ Speckle Suppression Index """
    sigma_r = np.std(filtered)
    r_dash = np.mean(filtered)

    sigma_f = np.std(noisy)
    f_dash = np.mean(noisy)
    
    SSI = (sigma_r * f_dash) / (r_dash * sigma_f)
    print(f'        [*] SSI @ {SSI}')
    
def ENL(filtered):
    """ Equivalent Number of Looks """
    mu = np.mean(filtered)
    sigma = np.var(filtered)

    ENL = (mu / sigma)**2
    print(f'        [*] ENL @ {ENL}')

def SMPI(noisy, filtered):
    """ Speckle Suppression and Mean Preservation Index """
    sigma_r = np.std(filtered)
    r_dash = np.mean(filtered)

    sigma_f = np.std(noisy)
    f_dash = np.mean(noisy)

    Q = 1 + abs(f_dash - r_dash)

    SMPI = Q * (sigma_r / sigma_f)
    print(f'        [*] SMPI @ {SMPI}')
    
def SSIM(img_n, img_f):
    SSIM = ssim(img_f, img_n, data_range=max(img_f.max(), img_n.max()) - min(img_f.min(), img_n.min()))
    print(f'        [*] SMPI @ {SSIM}')
    
def PSNR(img_n, img_f):
    PSNR = psnr(img_f, img_n, data_range=max(img_f.max(), img_n.max()) - min(img_f.min(), img_n.min()))
    print(f'        [*] PSNR @ {psnr}')
    

if __name__ == '__main__':
    noisy_files = glob((args.noisy_dir+'/*.'+args.ext).format('float32'))
    denoi_files = glob((args.filtered_dir+'/*.'+args.ext).format('float32'))
    print(f' [] Found {len(noisy_files)} files')
    print(f'[*] Computing Metrics')
    print(f'    [] Overridding PILs DecompressionBombError')
    Image.MAX_IMAGE_PIXELS = None  # Override PIL's DecompressionBombError
    
#     print(f'    [] Loading Noisy Image @ {args.noisy}')
#     img_n = np.asarray(Image.open(args.noisy).convert('L'))  # Noisy
#     print(f'    [] Loading Filtered Image @ {args.filtered}')
#     img_f = np.asarray(Image.open(args.filtered).convert('L'))  # Filtered
#     print(f'[*] Starting Metrics Computation')
    
    for idx in range(len(noisy_files)):
        print(f'    [] Loading Noisy Image @ {noisy_files[idx]}')
        img_n = np.asarray(Image.open(noisy_files[idx]).convert('L'))  # Noisy
        print(f'    [] Loading Filtered Image @ {denoi_files[idx]}')
        img_f = np.asarray(Image.open(denoi_files[idx]).convert('L'))  # Filtered
        
    ###########################################################################
        print(f'      [*] Starting Metrics Computation')
        SSI(img_n, img_f)
        ENL(img_f)
        SMPI(img_n, img_f)
        SSIM(img_n, img_f)
        PSNR(img_n, img_f)
    
    ###########################################################################
    print(f'[*] Script Succeeded')