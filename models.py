from keras.layers import Conv2D, Dense, BatchNormalization, Input, \
    MaxPooling2D, GlobalMaxPooling2D, dot
from keras.models import Model


def siamese_model(tower, img_size):
    """
    Turns a processing tower into a siamese model
    that uses the cosine-similarity as the distance metric.
    """
    img_a = Input(img_size + (3,))
    img_b = Input(img_size + (3,))

    # Pass the inputs through the towers
    w = tower(img_a)
    w = GlobalMaxPooling2D()(w)
    v = tower(img_b)
    v = GlobalMaxPooling2D()(v)

    # Merge the outputs into one
    distance = dot([u, v], axes=1, normalize=True)

    model = Model(inputs=[input_a, input_b], outputs=distance)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def simple_tower():
    """
    A simple 3 layer convolutional tower for processing
    images.
    """
    # None means variable sized input
    img = Input((None, None, 3))
    x = img

    # Define the shared weight tower
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    return Model(inputs=img, outputs=x)
