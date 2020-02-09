from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model


def cnn_model(size) -> Model:
    inputs = Input(shape=(size, size, 1))
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(size, size, 1))(inputs)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.15)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.15)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(rate=0.3)(x)

    head_root = Dense(168, activation='softmax', name='root')(x)
    head_vowel = Dense(11, activation='softmax', name='vowel')(x)
    head_consonant = Dense(7, activation='softmax', name='consonant')(x)

    return Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])


if __name__ == "__main__":
    simple_model = cnn_model(64)
    simple_model.summary()
