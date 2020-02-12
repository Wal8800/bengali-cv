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


def kaggle_cnn_model(size) -> Model:
    inputs = Input(shape=(size, size, 1))

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu',
                   input_shape=(size, size, 1))(inputs)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Flatten()(model)
    model = Dense(1024, activation="relu")(model)
    model = Dropout(rate=0.3)(model)
    dense = Dense(512, activation="relu")(model)

    head_root = Dense(168, activation='softmax', name='root')(dense)
    head_vowel = Dense(11, activation='softmax', name='vowel')(dense)
    head_consonant = Dense(7, activation='softmax', name='consonant')(dense)

    return Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])


if __name__ == "__main__":
    simple_model = kaggle_cnn_model(128)
    simple_model.summary()
