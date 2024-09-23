import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from glob import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_npy(filepath):
    image = np.load(filepath)
    image = np.array(image).astype('float32')
    mean = np.mean(image)
    var = np.var(image)
    return [image, mean, var]

def normalize_image(image):
    image_mean = np.mean(image)
    image_std = np.std(image)
    if image_std == 0:
        normalized_image = image - image_mean
    else:
        normalized_image = (image - image_mean) / image_std
    return normalized_image

def calculate_metrices(fd_images, qd_images, result_images, win_size=None):
    qd_psnr_list = []
    qd_ssim_list = []
    result_psnr_list = []
    result_ssim_list = []

    for i in range(len(fd_images)):

        print(str(i))
        fd_image = np.squeeze(normalize_image(fd_images[i]))
        qd_image = np.squeeze(normalize_image(qd_images[i]))
        result_image = np.squeeze(normalize_image(result_images[i]))

        # Calculating PSNR and SSIM metrics for quarter dose images
        qd_psnr = psnr(fd_image, qd_image, data_range=qd_image.max() - qd_image.min())
        qd_ssim = ssim(qd_image, fd_image, data_range=qd_image.max() - qd_image.min(), multichannel=True)
        qd_psnr_list.append(qd_psnr)
        qd_ssim_list.append(qd_ssim)

        # Calculating PSNR and SSIM metrics for result images
        result_psnr = psnr(fd_image, result_image, data_range=result_image.max() - result_image.min())
        result_ssim = ssim(result_image, fd_image, data_range=result_image.max() - result_image.min(), multichannel=True)
        result_psnr_list.append(result_psnr)
        result_ssim_list.append(result_ssim)


    metrics_df = pd.DataFrame({
        'qd_psnr': qd_psnr_list,
        'qd_ssim': qd_ssim_list,
        'result_psnr': result_psnr_list,
        'result_ssim': result_ssim_list
    })

    return metrics_df

# Defining the Paths
path_result = r"..\CNCL-denoising-main\result\test\result"
path_noisy = r"..\CNCL-denoising-main\data\mayo_data_pp\test\quarter_1mm"
path_clean = r"..\CNCL-denoising-main\data\mayo_data_pp\test\full_1mm"

noisy_list = sorted(glob(os.path.join(path_noisy, "*.npy")))
clean_list = sorted(glob(os.path.join(path_clean, "*.npy")))
result_list = sorted(glob(os.path.join(path_result, "*.npy")))

qd_images = [load_npy(i)[0] for i in noisy_list]
fd_images = [load_npy(i)[0] for i in clean_list]
result_images = [load_npy(i)[0] for i in result_list]

pd = calculate_metrices(fd_images, qd_images, result_images)

print(pd)
pd.to_csv(r"C:\Users\manta\OneDrive\Desktop\CT Scan Reconstruction\CNCL-denoising-main\Mycode\resultss.csv")

