# CT-Scan-Reconstruction

## Implementation of a Adversial Training, Deep Learning Framework -- Using Pytorch

> The architecture was inspired by [Content-Noise Complementary Learning for
Medical Image Denoising](https://www.researchgate.net/profile/Jin-Lujia/publication/354642580_Content-Noise_Complementary_Learning_for_Medical_Image_Denoising/links/62132d864be28e145ca63915/Content-Noise-Complementary-Learning-for-Medical-Image-Denoising.pdf)



## Overview
### Data

The Original Dataset was acquired from the AAPM Grand Challenge Data Repository.

The dataset includes 30 contrast-enhanced abdominal CT patient scans, each acquired in the portal venous phase using a Siemens SOMATOM Flash scanner. The data are deindentified and the case ID’s (Lxxx) are the same as were used in the Grand Challenge and subsequent publications. Data acquired at 120 kV and 200 quality reference mAs (QRM) are referred to as Full Dose (FD) data, and simulated data corresponding to 120 kV and 50 QRM are referred to as Quarter Dose Dose (QD) data.

The Link of the Dataset is : [Dataset Link](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h)

### Model

![image](https://github.com/Manty2503/CT-Scan-Reconstruction/assets/119813195/bf63a52d-0dcf-44c5-b1c2-ee56dfad3aed)

So this is the overall architecture CNCL Framework i.e. Content Noise Complementary Learning Framework.


**Generator** 
> It comprises 2 predictors branches which are organised in parallel way. With same network architecture. 
Noise Learning Branch: For Preserving Contrast and Structural Information,
Content Learning Branch: For Providing stable noise reduction

![image](https://github.com/Manty2503/CT-Scan-Reconstruction/assets/119813195/da31871b-a230-40bb-87a0-4653a1daad1c)

Both these branches have a Encoder-Decoder Architecture(U-Net) architecture as shown in the above image.

**The Fusion Operator**
> Concatenation + 1X1 Convolution Layer.

**Discriminator**
> PatchGAN, thereby comparing patches by patches.


### Training

The model is trained in a adveserial fashion where the Generator was responsible for generating Denoised Images whereas the Discrimator was assigned a job to discriminate between the original and the genrated images.

The model has been trained upto 120 epochs.

Loss Function - GAN Adversial Loss + L1 Loss for the Generated Image

Metrices - PSNR, SSIM

Optimizer - Adam


## How to Use

### Dependecies
This tutorial uses Tensorflow(2.11.0), OpenCV(cv2), Python(3.8).

You can modify this accordinally.


### Code Files Descriptions

MyDataLoader_test.py/MyDataloader_train.py - Inorder to load the files and create batches for the Pytorch DataLoaders.

Mymodel.py - Has the complete model structure defined in it.

Mytrain.py - To train the model and the weight file in .h5 format.

Mytest.py - To generate the Denoised images.

Myevaluation.py - To calculate the model performance.


### Results


![image](https://github.com/Manty2503/CT-Scan-Reconstruction/assets/119813195/5e63f78d-d10f-4cad-b15b-f829fe90a452)






# CT-Scan-Reconstruction
