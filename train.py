from image_segmentation import keras_segmentation


def train(image_folder: str, annotation_folder: str, output_model_path: str, n_epochs: int, input_height: int,
          input_width: int, n_classes: int):

    # Case 1: Use VGG
    # model = keras_segmentation.models.unet.vgg_unet(n_classes=n_classes,  input_height=input_height, input_width=input_width)

    # Case 2: Use ResNet
    model = keras_segmentation.models.unet.resnet50_unet(n_classes=n_classes, input_height=input_height,
                                                         input_width=input_height)

    model.train(
        train_images=image_folder,
        train_annotations=annotation_folder,
        checkpoints_path=output_model_path, epochs=n_epochs
    )


if __name__ == "__main__":

    image_folder = 'data/train/images'
    annotation_folder = 'data/train/annotations'
    output_model_path = 'models/cccd'
    n_epochs = 30
    input_height = 608
    input_width = 608

    # number of classes = number of target classes + background
    n_classes = 2

    train(image_folder, annotation_folder, output_model_path, n_epochs, input_height, input_width, n_classes)
