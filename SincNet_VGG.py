def model_generator():
    
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.applications.vgg19 import VGG19
    from keras.callbacks import ModelCheckpoint
    from keras.layers import Reshape
    
    
    inp =  Input(shape=(16000,1))
    
    x = sincnet.SincConv1D(227, 251, 16000)(inp)
    
    x = MaxPooling1D(pool_size=70)(x)
    
    x = sincnet.LayerNorm()(x)
    
    reshape_layer=Reshape((225, 227,-1))(x)
    
    x=VGG19(include_top=False, weights=None, 
            input_tensor=reshape_layer, input_shape=(232, 227,-1), pooling=None)
    
    x = Flatten()(x.output)
    
    x = Dense(4096,activation="relu")(x)
    
    dense4 = Dense(10,activation=None)(x)
    
    model = Model(inputs=inp, outputs=x)
    
    model.compile(loss=wrapped_partial(K.categorical_crossentropy, from_logits=True),
          optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
          ,metrics=['accuracy'])
    
    return model
