import os, argparse
import cv2, spacy, numpy as np

from keras.optimizers import SGD
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
K.set_image_data_format('channels_last')

from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model


# File paths for the model, all of these except the CNN Weights are
# provided in the repo, See the models/CNN/README.md to download VGG weights
VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights.h5'

# Change the value of verbose to 0 to avoid printing the progress statements
verbose = 1


def get_image_model():
    """
    Inception_V3 model update
    """

    # from models.CNN.VGG import VGG_16
    # image_model = VGG_16(CNN_weights_file_name)
    from keras.applications import InceptionV3

    base_model = InceptionV3(weights='imagenet',
                             include_top=False, input_shape=(299, 299, 3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(4096, activation='relu')(x)
    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model


def get_image_features(image_file_name, CNN_weights_file_name):
    """
    Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector

    :param image_file_name: test image file name (including path)
    :param CNN_weights_file_name: CNN weight file name (including path)
    :return: image vector
    """

    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (299, 299))
    im = im.astype(np.float32, copy=False)
    print('im: {}'.format(im.shape))
    # The mean pixel values are taken from the VGG authors,
    # which are the values computed from the training dataset.
    # BGR?
    # mean_pixel = [103.939, 116.779, 123.68]

    # im = im.astype(np.float32, copy=False)
    # for c in range(3):
    #     im[:, :, c] = im[:, :, c] - mean_pixel[c]

    # convert the image to RGBA?
    # NCHW or WHCN (Width / Height / Color / Number of pic)
    # im = im.transpose((2,0,1))

    print('im: {}'.format(im.shape))


    # this axis dimension is required because VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)

    image_features[0,:] = get_image_model().predict(im)[0]
    return image_features


def get_VQA_model(VQA_weights_file_name):
    """
    Given the VQA model and its weights, compiles and returns the model

    :param VQA_weights_file_name: VQA model weight file (including path)
    :return: pre-trained VQA model
    """
    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)

    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model


def get_question_features(question):
    """
    For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector

    :param question: question string
    :return: question vector
    """

    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def main():
    """
    accepts command line arguments for image file and the question and
    builds the image model (VGG) and the VQA model (LSTM and MLP)
    prints the top 5 response along with the probability of each

    :return: None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-image_file_name', type=str, default='test.jpg')
    parser.add_argument('-question', type=str, default='What vehicle '
                                                       'is in the picture?')
    args = parser.parse_args()

    if verbose : print("\n\n\nLoading image features ...")
    image_features = get_image_features(args.image_file_name, CNN_weights_file_name)

    if verbose : print("Loading question features ...")
    question_features = get_question_features(str(args.question))

    if verbose : print("Loading VQA Model ...")
    vqa_model = get_VQA_model(VQA_weights_file_name)

    if verbose : print("\n\n\nPredicting result ...")
    y_output = vqa_model.predict([question_features, image_features])
    y_sort_index = np.argsort(y_output)

    # This task here is represented as a classification into a 1000 top answers
    # this means some of the answers were not part of training and thus would
    # not show up in the result.
    # These 1000 answers are stored in the sk-learn Encoder class
    labelencoder = joblib.load(label_encoder_file_name)
    for label in reversed(y_sort_index[0,-5:]):
        print(round(y_output[0, label]*100, 2), "% ",
              labelencoder.inverse_transform(label))

if __name__ == "__main__":
    main()
