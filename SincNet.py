def model_generator():
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    import sincnet

    inp = Input(shape=(50999, 1))

    x = sincnet.SincConv1D(80, 251, 16000)(inp)

    x = MaxPooling1D(pool_size=3)(x)

    x = sincnet.LayerNorm()(x)

    # x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(60, 5, strides=1, padding='valid', kernel_initializer=glorot_normal(seed=None))(x)

    x = MaxPooling1D(pool_size=3)(x)

    x = sincnet.LayerNorm()(x)

    # x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(60, 5, strides=1, padding='valid', kernel_initializer=glorot_normal(seed=None))(x)

    x = sincnet.LayerNorm()(x)

    # x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)

    x = Dense(128, kernel_initializer=glorot_normal(seed=None))(x)
    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(64, kernel_initializer=glorot_normal(seed=None))(x)
    x = Dropout(0.5)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.2)(x)

    dense4 = Dense(10, activation=None)(x)

    model = Model(inputs=inp, outputs=dense4)

    model.compile(loss=wrapped_partial(K.categorical_crossentropy, from_logits=True),
                  optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
                  , metrics=['accuracy'])

    return model
