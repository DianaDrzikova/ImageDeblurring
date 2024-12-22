# ImageDeblurring

Compile:
```
mkdir build
cd build
cmake ../src
make
```
Navigate to /bin and run:
```
./stochastic_deconvolution_sample -i <input_image_path> [-r <regularizer_type>]
```
regularizer types:
```
tv, gamma, combination, sparse, data, discontinuous
```

Results will be stored in **\bin**.

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