# ImageDeblurring

Implementation of non-blind image deconvolution using stochastic random walks, applying arbitrary priors, including non-convex and data-dependent regularizers.

## Short Description

The implementation presented in this work is based on the Stochastic Deconvolution framework introduced by Gregson et al. in their 2013 paper. This novel approach addresses the non-blind image deconvolution problem using stochastic random walks, applying arbitrary priors, including non-convex and data-dependent regularizers.

We built on the code snippet provided by the authors. To enhance the method’s usability and adaptability, we reimplemented the algorithm using OpenCV, a modern and widely adopted image-processing library.

We expanded the original framework by implementing missing features mentioned in the
paper, such as:

    • Handling boundary conditions and saturation regions
    • Color processing
    • Gamma correction
    • Integration of three additional regularization terms

To validate our implementation, we designed an evaluation pipeline using the Jupyter Notebook. We evaluated our implementation using traditional metrics such as PSNR and SSIM to measure its performance.

Finally, we created and shared a dataset that includes blurred images, ground truth, and blur kernels. We developed this dataset to address the difficulty of obtaining such datasets for non-blind deblurring.

## Acknowledgements

 - [Implemented according to the paper by Gregson, J., Heide, F., Hullin, M. B., Rouf, M., and Heidrich, W. Stochastic Deconvolution. In IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) (June 2013)](https://www.cs.ubc.ca/labs/imager/tr/2013/StochasticDeconvolution/)

## Usage

### Compilation

```
mkdir build
cd build
cmake ../src
make
```

### Execution

Navigate to `/bin` and run:
```
./stochastic_deconvolution_sample [-i <input_image>] [-r <regularizer_type>]
```

- ```input_image``` is the image input for the deconvolution,
- ```regularizer_type``` is the type of the regularizer you want to use for the refinement of the deblurring process (e.g. tv, gamma, combination, sparse, data, discontinuous).

Results will be stored in the `/bin` folder.

### Dataset script

```
python blur_script.py \
  --train_input_dir ../dataset/images/train \
  --test_input_dir ../dataset/images/test \
  --train_output_blurred_dir ../dataset/blurred_images/train \
  --test_output_blurred_dir ../dataset/blurred_images/test \
  --train_output_labels_dir ../dataset/labels/train \
  --test_output_labels_dir ../dataset/labels/test \
  --random_psf
```

## Examples Of The Deconvolution

### Ground Truth Image

![Ground Truth Image](https://github.com/DianaDrzikova/ImageDeblurring/blob/main/out_images/dandelion.jpg)

### Image Before Deconvolution - Blurred Image, input (Kernel, 3x3 uniform 1/9)
![Blurred Image, input (Kernel, 3x3 uniform 1/9)](https://github.com/DianaDrzikova/ImageDeblurring/blob/main/out_images/blurred.png)

### Image After Deconvolution - Total Variation Regularizer

![Total Variation Regularizer](https://github.com/DianaDrzikova/ImageDeblurring/blob/main/out_images/results/tv_intrinsic.png)

### Image After Deconvolution - Data-dependent Regularizer

![Data-dependent Regularizer](https://github.com/DianaDrzikova/ImageDeblurring/blob/main/out_images/results/data_intrinsic.png)
