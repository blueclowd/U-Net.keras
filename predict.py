
import numpy as np
import cv2
from image_segmentation import keras_segmentation
from utils import visualize

def predict(model, image: np) -> np:
    """
    Segment image using the segmentation model
    :param model: segmentation model
    :param image: image to be segmented
    :return: labeled image
    """

    label_img = model.predict_segmentation(inp=image)

    output_height, output_width = model.output_height, model.output_width
    img_height, img_width = image.shape[:2]

    labels = np.unique(label_img)
    labels = labels[labels != 0]

    resized_label_img = np.zeros((img_height, img_width), np.uint8)
    max_contours = []

    # Collect maximal blobs for each label
    for label in labels:

        mask_img = np.zeros((output_height, output_width), np.uint8)
        mask_img[label_img == label] = 255

        contours, _ = cv2.findContours(mask_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(c) for c in contours]

        if areas:

            max_index = np.argmax(areas)
            max_cnt = contours[max_index]

            max_contours.append((label, max_cnt, int(cv2.contourArea(max_cnt))))

    # Resize the output of the model to the original size
    for max_contour in max_contours:

        label, contour, _ = max_contour

        # Recover the contour
        recovered_max_contour = contour.copy()
        recovered_max_contour[:, :, 0] = contour[:, :, 0] * img_width / output_width
        recovered_max_contour[:, :, 1] = contour[:, :, 1] * img_height / output_height

        cv2.drawContours(resized_label_img, [recovered_max_contour], -1, color=int(label), thickness=-1)

    return resized_label_img


if __name__ == "__main__":

    model = keras_segmentation.predict.model_from_checkpoint_path('models/cccd')

    test_img = cv2.imread('data/test/20190719102257_45585446.jpg')

    label_img = predict(model, test_img)

    color_map = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

    cv2.imshow('Label', visualize(label_img, color_map))
    cv2.waitKey(0)
