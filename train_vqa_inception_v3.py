
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers.con
import numpy as np
import cv2
import os


# If you want to use a GPU set its index here
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# This function comes from Google's ImageNet Preprocessing Script
def central_crop(image, central_fraction):
    """Crop the central region of the image.
    Remove the outer parts of an image but retain the central region of the image
    along each dimension. If we specify central_fraction = 0.5, this function
    returns the region marked with "X" in the below diagram.
       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where "X" is the central 50% of the image.
       --------
    Args:
    image: 3-D array of shape [height, width, depth]
    central_fraction: float (0, 1], fraction of size to crop
    Raises:
    ValueError: if central_crop_fraction is not within (0, 1].
    Returns:
    3-D array
    """
    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    depth = img_shape[2]
    fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
    bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
    bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

    bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
    bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

    image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]
    return image


def get_processed_image(img_path):
    # Load image and convert from BGR to RGB

    # im = cv2.resize(cv2.imread(img_path), (299, 299))

    im = np.asarray(cv2.imread(img_path))[:,:,::-1]
    im = central_crop(im, 0.875)
    im = cv2.resize(im, (299, 299))

    im = inception_v4.preprocess_input(im)
    if K.image_data_format() == "channels_first":
        im = np.transpose(im, (2,0,1))
        im = im.reshape(-1,3,299,299)
    else:
        im = im.reshape(-1,299,299,3)
    return im



# model_inception_v4 = inception_v4.create_model(weights='imagenet', include_top=False)
from keras.applications import InceptionV3
model_inception_v3_notop = InceptionV3(weights='imagenet', include_top=False)
# model_inception_v3 = InceptionV3(weights='imagenet', include_top=True)
# print("notop", model_inception_v3_notop.layers)
print("model.output: ", model_inception_v3_notop.output)
print(model_inception_v3_notop.summary())

# add a global spatial average pooling layer
x = model_inception_v3_notop.output
x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
# let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.8)(x)
x = Dense(4096, activation='relu')(x)
model = Model(input=model_inception_v3_notop.input, output=x)
print(model.summary())