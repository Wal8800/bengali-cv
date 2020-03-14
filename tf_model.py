import efficientnet.tfkeras as efn
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.layers import Input, Dense, Layer, BatchNormalization, Conv2D
from tensorflow.keras.models import Model


# https: // gist.github.com / digantamisra98 / 35ca0ec94ebefb99af6f444922fa52cd
class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class GeneralizedMeanPool2D(layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        # self.gm_exp = tf.Variable(3.0, dtype=tf.float32, trainable=True)
        self.gm_exp = self.add_weight(name=f"{name}_gm_exp", initializer=tf.keras.initializers.Constant(value=3.0),
                                      dtype=tf.float32,
                                      trainable=True)

    def call(self, x):
        return (tf.reduce_mean(tf.abs(x ** self.gm_exp), axis=[1, 2], keepdims=False) + 1.e-7) ** (
                1. / self.gm_exp)


def experiment_tail_block(x, name):
    # x = Mish()(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = GeneralizedMeanPool2D(name)(x)
    return x


def tail_block(x, name):
    # x = MishActivations()(x)
    # # x = Activation("relu")(x)
    # x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    # x = BatchNormalization()(x)

    x = GeneralizedMeanPool2D(name)(x)
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.3)(x)

    return x


def dense_net_121_model(input_shape) -> Model:
    inputs = Input(shape=input_shape)

    base_model = DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        input_shape=input_shape,
        pooling=None
    )

    a = tail_block(base_model.output, "root")
    b = tail_block(base_model.output, "vowel")
    c = tail_block(base_model.output, "consonant")

    head_root = Dense(168, activation='softmax', name='root')(a)
    head_vowel = Dense(11, activation='softmax', name='vowel')(b)
    head_consonant = Dense(7, activation='softmax', name='consonant')(c)

    return Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])


def dense_net_169(input_shape) -> Model:
    inputs = Input(shape=input_shape)

    base_model = DenseNet169(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        input_shape=input_shape
    )

    a = tail_block(base_model.output, "root")
    b = tail_block(base_model.output, "vowel")
    c = tail_block(base_model.output, "consonant")

    head_root = Dense(168, activation='softmax', name='root')(a)
    head_vowel = Dense(11, activation='softmax', name='vowel')(b)
    head_consonant = Dense(7, activation='softmax', name='consonant')(c)

    return Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])


def efficient_net_b3(input_shape) -> Model:
    inputs = Input(shape=input_shape)

    base_model = efn.EfficientNetB2(weights=None, include_top=False, input_tensor=inputs, pooling=None,
                                    classes=None)
    for layer in base_model.layers:
        layer.trainable = True

    # a = tail_block(base_model.output, "root")
    # b = tail_block(base_model.output, "vowel")
    # c = tail_block(base_model.output, "consonant")

    x = GeneralizedMeanPool2D('gem')(base_model.output)

    a = Dense(512)(x)
    b = Dense(512)(x)
    c = Dense(512)(x)

    head_root = Dense(168, activation='softmax', name='root')(a)
    head_vowel = Dense(11, activation='softmax', name='vowel')(b)
    head_consonant = Dense(7, activation='softmax', name='consonant')(c)

    return Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])


if __name__ == "__main__":
    example_model = efficient_net_b3((224, 224, 3))
    example_model.summary()
