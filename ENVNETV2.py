def model_generator():
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint

    inp = Input(shape=(50999, 1))

    # ----------------------
    conv1 = Conv1D(filters=32, kernel_size=64, strides=2, activation='relu')(inp)
    norm1 = BatchNormalization()(conv1)
    # ----------------------

    conv2 = Conv1D(filters=64, kernel_size=16, strides=2, activation='relu')(norm1)
    norm2 = BatchNormalization()(conv2)

    # ----------------------

    pool2 = MaxPool1D(pool_size=64, strides=64)(norm2)

    pool2 = Lambda(lambda x: permute_dimensions(x, (0, 2, 1)))(pool2)

    SWAP = Lambda(lambda x: expand_dims(x, axis=3), name="swap")(pool2)

    print(SWAP.shape)

    # ----------------------
    conv3 = Conv2D(filters=32, kernel_size=(8, 8), strides=(1, 1), activation='relu')(SWAP)
    norm3 = BatchNormalization()(conv3)

    # ----------------------
    conv4 = Conv2D(filters=32, kernel_size=(8, 8), strides=(1, 1), activation='relu')(norm3)
    norm4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(5, 3), strides=(5, 3), padding='valid')(norm4)
    # ----------------------

    conv5 = Conv2D(filters=64, kernel_size=(1, 4), strides=(1, 1), activation='relu')(pool4)
    norm5 = BatchNormalization()(conv5)
    # ----------------------

    conv6 = Conv2D(filters=64, kernel_size=(1, 4), strides=(1, 1), activation='relu')(norm5)
    norm6 = BatchNormalization()(conv6)
    pool6 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(norm6)
    # ----------------------

    conv7 = Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu')(pool6)
    norm7 = BatchNormalization()(conv7)

    pool8 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(norm7)
    # ----------------------

    conv9 = Conv2D(filters=256, kernel_size=(1, 2), strides=(1, 1), activation='relu')(pool8)
    norm9 = BatchNormalization()(conv9)

    conv10 = Conv2D(filters=256, kernel_size=(1, 2), strides=(1, 1), activation='relu')(norm9)
    norm10 = BatchNormalization()(conv10)
    pool10 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(norm10)
    # ----------------------

    flat = Flatten()(pool10)

    dense1 = Dense(4096, activation='relu')(flat)
    drop1 = Dropout(0.5)(dense1)

    dense2 = Dense(4096, activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)

    dense4 = Dense(10, activation=None)(drop2)

    model = Model(inputs=inp, outputs=dense4)

    model.compile(loss=wrapped_partial(K.categorical_crossentropy, from_logits=True),
                  optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
                  , metrics=['accuracy'])

    return model