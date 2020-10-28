def model_generator():
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    import scipy.io as sio
    
    contents_mat=sio.loadmat('/home/an80020/Desktop/for_mohsen/filters.mat')
    gammatone = contents_mat['filters']
    w_coch = np.transpose(gammatone)
    w_coch2=np.zeros((512,64))
    for i in reversed(range(64)):
        w_coch2[:,63-i]=w_coch[:,i]

    inp = Input(shape=(50999, 1))

    # ----------------------
    conv1 = Conv1D(filters=64, kernel_size=512, kernel_initializer=init.Constant(w_coch2),
                   activation='relu', trainable=False, name='filterbank')(inp)

    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D(pool_size=8, strides=8)(norm1)
    # ----------------------

    conv2 = Conv1D(filters=32, kernel_size=32, strides=2, activation='relu', padding='valid')(pool1)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPool1D(pool_size=8, strides=8)(norm2)

    # ----------------------
    conv3 = Conv1D(filters=64, kernel_size=16, strides=2, activation='relu', padding='valid')(pool2)
    norm3 = BatchNormalization()(conv3)

    # ----------------------
    conv4 = Conv1D(filters=128, kernel_size=8, strides=2, activation='relu', padding='valid')(norm3)
    norm4 = BatchNormalization()(conv4)

    # ----------------------
    conv5 = Conv1D(filters=256, kernel_size=4, strides=2, activation='relu', padding='valid')(norm4)
    norm5 = BatchNormalization()(conv5)
    pool3 = MaxPool1D(pool_size=4, strides=4)(norm5)

    flat = Flatten()(pool3)
    dense1 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02))(flat)
    drop2 = Dropout(.5)(dense1)
    dense2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(drop2)
    drop2 = Dropout(.25)(dense2)
    dense3 = Dense(10, activation=None)(drop2)
    model = Model(inputs=[inp], outputs=dense3)

    model.compile(loss=wrapped_partial(K.categorical_crossentropy, from_logits=True),
                  optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
                  , metrics=['accuracy'])

    return model
