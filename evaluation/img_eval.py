from math import log10, sqrt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cpbd

def calc_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def PSNR(original, compressed):
	mse = np.mean((original - compressed) ** 2)
	if(mse == 0):
		return float('inf')
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def main(total):
    fvalue=[0, 0, 0]
    for i in range(total*5):
        print(i)
        original = cv2.imread(f"PSNR/ref_pre1/{i}.jpg")
        generated = cv2.imread(f"PSNR/ref_pre2/{i}.jpg")
        g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
        psnr, ssim_score, cpbd =  PSNR(g1, g2), ssim(g1, g2), cpbd.compute(g2)
        fvalue[0] += psnr
        fvalue[1] += ssim_score
        fvalue[2] += cpbd
    print(fvalue)
	
if __name__ == "__main__":
	main(500)
