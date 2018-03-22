#Deep Learning in Super-resolution Reconstruction

---
#Paper Review

Super-resolution (SR), which aims at recovering high-resolution images (or videos) from a low-resolution one, is a classical problem in computer vision.It has important value in monitoring equipment, satellite imagery, and medical imaging.

SR is an inverse problem, since a multiplicity of solutions exist for any given low-resolution pixel. Such a problem is typically mitigated by constraining the solution space by strong prior information. In the traditional approach, this prior information can be learned through the example of several pairs of low-high-resolution images that appear. The SR based on deep learning directly learns the end-to-end mapping function of the resolution image to the high resolution image through the neural network.


The two commonly used indexes for quantitative evaluation of SR quality are Peak Signal-to-Noise Ratio (PSNR) and Structure Similarity Index (SSIM). The higher these two values ​​are, the closer the pixel values ​​of the reconstruction result are to the gold standard.

The followings are some classic deep learning SR methods.

##1. SRCNN

The super-resolution convolutional neural network was proposed earlier as a convolutional neural network for SR. The network structure is very simple, using only three convolutional layers.

![](https://pic4.zhimg.com/80/v2-82423d397a228f3b4522769ae9a85e83_hd.jpg)

This method first uses a bicubic interpolation for a low-resolution image to amplify it to the target size, and then uses a three-layer convolution network to do a nonlinear mapping. 

The following figure shows that at different magnifications, the SRCNN achieves better results than the traditional method.

![](https://pic1.zhimg.com/80/v2-a22efaa701ce06df06e2b3bf520609d3_hd.jpg)
![](https://pic1.zhimg.com/80/v2-b4ebbda16d8671d39abe06e6cad2d65e_hd.jpg)

In addition to the earliest use of CNN in SR problems, the author interprets the structure of the three-level convolution into three steps corresponding to the traditional SR method: image block extraction and feature representation, feature nonlinear mapping, and final reconstruction. 

##2. DRCN

The SRCNN has fewer layers and therefore, smaller field of vision(13x13). DRCN (Deeply-Recursive Convolutional Network for Image Super-Resolution, CVPR 2016, code) proposes to use more convolutional layers to increase the network receptive field (41x41), and to avoid excessive network parameters, this article proposes to use recursive neural networks (RNN). The basic structure of the network is as follows:

![](https://pic1.zhimg.com/80/v2-cb1c3003163537f45d5a2ab532d7c4db_hd.jpg)

![](https://pic4.zhimg.com/80/v2-c40759b372c4aa962a762d1db470da06_hd.jpg)

##3. ESPCN

In SRCNN and DRCN, low-resolution images are first obtained by up-sampling and interpolating to obtain the same size as high-resolution images. As a network input, convolution operation is performed at a higher resolution compared to Calculating convolutions on low-resolution images can reduce efficiency. ESPCN (Real-Time
Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network, CVPR 2016, code) proposes an efficient method for calculating convolutions directly on low resolution images to get high resolution images.

The core concept of ESPCN is the sub-pixel convolutional layer. As shown in the above figure, the network input is the original low-resolution image. After passing through two convolution layers, the feature image size obtained is the same as the input image, but the feature channel is r^2 (r is the target magnification of the image). Reorders the r^2 channels of each pixel into an rxr region, corresponding to a rxr-sized sub-block in the high-resolution image, so that the feature images of size r^2xHxW are rearranged into 1 x rH x rW size high resolution image. Although this transformation is called sub-pixel convolution, there is actually no convolution operation.
By using sub-pixel convolution, the process of image enlargement from low resolution to high resolution, the interpolation function is implicitly contained in the preceding convolution layer and can be learned automatically. Only at the last level, the size of the image is transformed. The previous convolution operation is efficient because it is performed on a low-resolution image.

![](https://pic4.zhimg.com/80/v2-9978df0775ec4be45a2894ce6d853e3c_hd.jpg)

![](https://pic1.zhimg.com/80/v2-eb45d86cad81d34f451797171903bc5e_hd.jpg)

##4. VESPCN

In the SR problem of video images, adjacent frames have a strong correlation. The above methods are only processed on a single image, and VESPCN (
Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation, arxiv 2016) proposes the use of time-series images in video for high-resolution reconstruction and can meet the efficiency requirements of real-time processing. The schematic diagram of the method is as follows, mainly including three aspects:

One is to correct the offset of the adjacent frame, that is, to estimate the displacement by Motion estimation first, and then use the displacement parameter to spatially transform adjacent frames to align the two. The second is to stack several adjacent frames after alignment. As a three-dimensional data, three-dimensional convolution is used on low-resolution three-dimensional data. The result size is r^2\times H\times W. The third is to use the idea of ​​ESPCN to rearrange the convolution results to get a high-resolution image of size 1\times rH\times rW.
The process of Motion estimation can be calculated by the traditional optical flow algorithm. DeepMind proposed a Spatial Transformer Networks to estimate the spatial transform parameters through CNN. VESPCN uses this method and uses multi-scale Motion estimation to obtain an initial transformation at a lower resolution than the input image, and to obtain more accurate results at the same resolution as the input image, as shown in the following figure:

##5. SRGAN
SRGAN (Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, arxiv, 21 Nov, 2016) will use a generative countermeasure network (GAN) for the SR problem. The starting point is that the traditional method generally deals with a smaller magnification. When the magnification of the image is above 4, it is easy to make the result appear to be too smooth and lack some realism in details. Therefore SRGAN uses GAN to generate details in the image.

The cost function used by the traditional method is generally the minimum mean square error (MSE), ie

---
#Implementation
using Python 3.6.4, Tensorflow 1.1.0, Pytorch 0.3.1, cuda 8.0, cudnn 5.1

##1. Video Processing
Cut video : ffmpeg -ss 00:00:30.0 -i input.avi -c copy -t 00:00:10.0 output.avi (https://superuser.com/questions/138331/using-ffmpeg-to-cut-up-video)

Split video into images : (Test)$ ffmpeg -i file.avi -r 50/1 $output%03d.bmp (https://stackoverflow.com/questions/10957412/fastest-way-to-extract-frames-using-ffmpeg)

Convert a set of images into a video : (sample)$ ffmpeg -r 50 -i test_image%03d.png -vcodec libx264 -y -an video.avi -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" (https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2)

##2.Test Pytorch Model
(video-super-resolution)$ python super_resolve.py --model [model_name]

##3. Train or Test Tensorflow Model 
Train module : (video-super-resolution)$ python main.py --stride=14 --is_train=True

Test module : (video-super-resolution)$ python main.py --stride=21 --is_train=False

