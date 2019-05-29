def model_generator():
    # Model Li et al.

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    from keras.layers.core import Lambda
    from keras.backend import permute_dimensions
    from keras.backend import expand_dims

    inp = Input(shape=(20480, 1))

    # ----------------------
    conv1 = Conv1D(filters=40, kernel_size=8, strides=1, activation='relu', padding='valid')(inp)

    conv2 = Conv1D(filters=40, kernel_size=8, strides=1, activation='relu', padding='valid')(conv1)

    pool1 = MaxPool1D(pool_size=128, strides=128)(conv2)

    # ----------------------

    swap = Lambda(lambda x: permute_dimensions(x, (0, 2, 1)), name="premute")(pool1)

    print(swap.shape)

    swap = Lambda(lambda x: expand_dims(x, axis=3), name="swap")(swap)

    conv_2d = Conv2D(filters=24, kernel_size=(6, 6), strides=(1, 1), activation='relu')(swap)

    conv_2d = Conv2D(filters=24, kernel_size=(6, 6), strides=(1, 1), activation='relu')(conv_2d)

    conv_2d = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(conv_2d)

    conv_2d = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(conv_2d)

    conv_2d = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu')(conv_2d)

    flat = Flatten()(conv_2d)

    dense1 = Dense(200, activation='relu')(flat)

    dense2 = Dense(10, activation='softmax')(dense1)

    model = Model(inp, dense2)

    model.compile(loss='mean_squared_logarithmic_error',
                  optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
                  , metrics=['accuracy'])

    return model