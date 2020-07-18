# Semantic Segmentation

#### Source repository: *[link](https://github.com/divamgupta/image-segmentation-keras)*
This repository works as a wrapper of the work above. Specific set of function parameters are exposed for general-purpose segmentation problems. To access the full capability of the work, please refer to the original document at /image_segmentation/README.md 

#### Prepare customized dataset for training
You need to make two folders

- Images - For all the training images
- Annotations - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

> It is suggested to work on square image for better performance

#### Models
Download and place under *./models*

#### Training
`python train.py`

#### Testing
`python predict.py`