# U-Net.keras

### Source repository: *[link](https://github.com/divamgupta/image-segmentation-keras)*
This repository works as a wrapper of the work above. Specific set of function parameters are exposed for general-purpose segmentation problems. To access the full capability of the work, please refer to the original document at /image_segmentation/README.md 

### Environment setup:
`pip install -r requirements.txt`

### Prepare customized dataset for training
Two folders are needed:

- images - For all the training images
- annotations - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

> It is suggested to work on square image for better performance

### Models
Download and place under *./models*

### Training
Adjust related parameters and run
`python train.py`

### Testing
Adjust related parameters and run
`python predict.py`