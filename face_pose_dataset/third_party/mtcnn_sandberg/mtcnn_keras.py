""" My implementation of MTCNN

    References:

        - [Effective TensorFlow 2  |  TensorFlow Core](https://www.tensorflow.org/guide/effective_tf2#recommendations_for_idiomatic_tensorflow_20)
        - [Migrate your TensorFlow 1 code to TensorFlow 2  |  TensorFlow Core](https://www.tensorflow.org/guide/migrate#saving_loading)
        - [mtcnn/factory.py at master · ipazc/mtcnn · GitHub](https://github.com/ipazc/mtcnn/blob/master/mtcnn/network/factory.py)
        - [facenet/detect_face.py at master · davidsandberg/facenet · GitHub](https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py)
        - [tf.keras.layers.Layer  |  TensorFlow Core v2.1.0](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)
        - [Microsoft Word - SPL_final_double.docx - spl.pdf](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)
        - [MTCNN_face_detection_alignment/det1.prototxt at master · kpzhang93/MTCNN_face_detection_alignment · GitHub](https://github.com/kpzhang93/MTCNN_face_detection_alignment/blob/master/code/codes/MTCNNv1/model/det1.prototxt)
        - [Convolutional Layers - Keras Documentation](https://keras.io/layers/convolutional/#conv2d)
        - [Understanding Input Output shapes in Convolution Neural Network | Keras](https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca)
        - [How Does A Face Detection Program Work? (Using Neural Networks)](https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff)
        - [What Does A Face Detection Neural Network Look Like?](https://towardsdatascience.com/face-detection-neural-network-structure-257b8f6f85d1)
        - [keras-mtcnn/keras_24net_v2.py at master · xiangrufan/keras-mtcnn · GitHub](https://github.com/xiangrufan/keras-mtcnn/blob/master/training/keras_24net_v2.py)

"""

import logging
import os
import sys
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

# REF: https://www.tensorflow.org/guide/gpu#allowing_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


conv_group = 1


def PNet(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=10, kernel_size=(3, 3), name="conv1", dtype="float64")(
        inputs
    )
    x = layers.PReLU(shared_axes=[1, 2], name="PReLU1", dtype="float64")(x)
    x = layers.MaxPool2D((2, 2), (2, 2), dtype="float64")(x)

    x = layers.Conv2D(16, (3, 3), name="conv2", dtype="float64")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="PReLU2", dtype="float64")(x)

    x = layers.Conv2D(32, (3, 3), name="conv3", dtype="float64")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="PReLU3", dtype="float64")(x)

    # Network splits after PReLU3
    output1 = layers.Conv2D(2, (1, 1), padding="same", name="conv4-1", dtype="float64")(
        x
    )
    output1 = layers.Softmax(3, name="prob", dtype="float64")(output1)

    output2 = layers.Conv2D(
        4, (1, 1), (1, 1), padding="same", name="conv4-2", dtype="float64"
    )(x)

    return models.Model(inputs=[inputs], outputs=[output2, output1], name="PNet")


def RNet(input_shape=[24, 24, 3]):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=28, kernel_size=(3, 3), name="conv1", dtype="float64")(
        inputs
    )
    x = layers.PReLU(shared_axes=[1, 2], name="prelu1", dtype="float64")(x)
    x = layers.MaxPool2D((3, 3), (2, 2), padding="same", name="pool1", dtype="float64")(
        x
    )

    x = layers.Conv2D(48, (3, 3), name="conv2", dtype="float64")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="prelu2", dtype="float64")(x)
    x = layers.MaxPool2D((3, 3), (2, 2), name="pool2", dtype="float64")(x)

    x = layers.Conv2D(64, (2, 2), name="conv3", dtype="float64")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="prelu3", dtype="float64")(x)

    x = layers.Flatten(dtype="float64")(x)
    x = layers.Dense(128, name="conv4", dtype="float64")(x)
    x = layers.PReLU(name="prelu4", dtype="float64")(x)

    output1 = layers.Dense(2, name="conv5-1", dtype="float64")(x)
    output1 = layers.Softmax(1, name="prob", dtype="float64")(output1)

    output2 = layers.Dense(4, name="conv5-2", dtype="float64")(x)

    return models.Model(inputs=[inputs], outputs=[output2, output1], name="RNet")


def ONet(input_shape=[48, 48, 3]):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        filters=32, kernel_size=(3, 3), strides=(1, 1), name="conv1", dtype="float64"
    )(inputs)
    x = layers.PReLU(shared_axes=[1, 2], name="prelu1", dtype="float64")(x)
    x = layers.MaxPool2D((3, 3), (2, 2), padding="same", name="pool1", dtype="float64")(
        x
    )

    x = layers.Conv2D(64, (3, 3), (1, 1), name="conv2", dtype="float64")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="prelu2", dtype="float64")(x)
    x = layers.MaxPool2D((3, 3), (2, 2), name="pool2", dtype="float64")(x)

    x = layers.Conv2D(64, (3, 3), (1, 1), name="conv3", dtype="float64")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="prelu3", dtype="float64")(x)
    x = layers.MaxPool2D((2, 2), (2, 2), padding="same", name="pool3", dtype="float64")(
        x
    )

    x = layers.Conv2D(128, (2, 2), name="conv4", dtype="float64")(x)
    x = layers.PReLU(shared_axes=[1, 2], name="prelu4", dtype="float64")(x)

    x = layers.Flatten(dtype="float64")(x)
    x = layers.Dense(256, name="conv5", dtype="float64")(x)
    x = layers.PReLU(name="prelu5", dtype="float64")(x)

    output1 = layers.Dense(2, name="conv6-1", dtype="float64")(x)
    output1 = layers.Softmax(1, name="prob", dtype="float64")(output1)

    output2 = layers.Dense(4, name="conv6-2", dtype="float64")(x)

    output3 = layers.Dense(10, name="conv6-3", dtype="float64")(x)

    return models.Model(
        inputs=[inputs], outputs=[output2, output3, output1], name="ONet"
    )


def load_nnet(model_func: Callable, in_file: str, out_file: Optional[str] = None):
    data_dict: dict = np.load(in_file, encoding="latin1", allow_pickle=True).item()
    model: models.Model = model_func()

    for key, value in data_dict.items():
        logging.debug("Loading %s/%s", model.name, key)
        layer: layers.Layer = model.get_layer(key)

        data: dict = data_dict[key]

        weights = []

        if "weights" in value:
            weights.append(value["weights"])
            logging.debug("%s/%s: Weights loaded.", model.name, key)

        if "biases" in value:
            weights.append(value["biases"])
            logging.debug("%s/%s: Biases loaded.", model.name, key)

        if "alpha" in value:
            weights.append(value["alpha"])
            logging.debug("%s/%s: Alpha loaded.", model.name, key)

        assert len(weights) > 0

        assert len(layer.get_weights()) == len(weights)

        for idx, (i, j) in enumerate(zip(layer.get_weights(), weights)):
            logging.debug(
                "%s/%s: Shapes loaded %s, and %s.", model.name, key, i.shape, j.shape
            )
            if i.shape != j.shape:
                j = j.reshape(i.shape)
                weights[idx] = j

        layer.set_weights(weights)

    if out_file is not None:
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        models.save_model(model, out_file)

    return model


def test_net():
    K.set_learning_phase(False)

    pnet = PNet()
    rnet = RNet()
    onet = ONet()

    test_data = tf.ones(shape=(3, 48, 48, 3))
    res1 = onet(test_data)
    # res2 = rnet(test_data)

    print(res1)


def generate_models(
    model_root, save=False,
):
    pnet = load_nnet(
        PNet,
        os.path.join(model_root, "det1.npy"),
        os.path.join(model_root, "PNET") if save else None,
    )

    rnet = load_nnet(
        RNet,
        os.path.join(model_root, "det2.npy"),
        os.path.join(model_root, "RNET") if save else None,
    )

    onet = load_nnet(
        ONet,
        os.path.join(model_root, "det3.npy"),
        os.path.join(model_root, "ONET") if save else None,
    )

    return pnet, rnet, onet


if __name__ == "__main__":
    logging.info("Python %s on %s" % (sys.version, sys.platform))
    sys.path.extend(["/home/sam/Workspace/projects/4-ImageGathering/face_pose_dataset"])
    logging.getLogger().setLevel(logging.DEBUG)
    generate_models()
