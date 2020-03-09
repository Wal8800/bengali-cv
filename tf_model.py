import efficientnet.tfkeras as efn
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169
from tensorflow.keras.layers import Input, Dropout, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects
from tensorflow_core.python.keras.layers import Conv2D


# https://github.com/digantamisra98/Mish/blame/master/Mish/TFKeras/mish.py
class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({'Mish': Mish(mish)})


class MishActivations(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))


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


def conv_block(x):
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)

    return x


def tail_block(x, name):
    # x = MishActivations()(x)
    # # x = Activation("relu")(x)
    # x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
    # x = BatchNormalization()(x)

    return GeneralizedMeanPool2D(name)(x)


def dense_net_121_model(input_shape) -> Model:
    inputs = Input(shape=input_shape)

    base_model = DenseNet121(
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

    base_model = efn.EfficientNetB0(weights=None, include_top=False, input_tensor=inputs, pooling=None,
                                    classes=None)
    for layer in base_model.layers:
        layer.trainable = True

    # a = tail_block(base_model.output, "root")
    # b = tail_block(base_model.output, "vowel")
    # c = tail_block(base_model.output, "consonant")

    x = GeneralizedMeanPool2D('gem')(base_model.output)
    x = Dropout(0.1)(x)
    x = Dense(512, activation='elu')(x)
    x = Dropout(0.1)(x)

    head_root = Dense(168, activation='softmax', name='root')(x)
    head_vowel = Dense(11, activation='softmax', name='vowel')(x)
    head_consonant = Dense(7, activation='softmax', name='consonant')(x)

    return Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])


if __name__ == "__main__":
    example_model = dense_net_169((128, 128, 3))
    example_model.summary()
