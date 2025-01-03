from keras import Sequential, Input
from keras.models import Model
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.src.optimizers import Adam


def build_model(config, input_shape, num_classes):
    inp = Input(shape=input_shape)

    # Layer-1: Conv2D-Conv2D-Pooling
    x = Conv2D(filters=config['filters'], kernel_size=config['kernel_1'], activation=config['activation'], padding=config['padding'])(inp)
    x = Conv2D(filters=config['filters'], kernel_size=config['kernel_1'], activation=config['activation'], padding=config['padding'])(x)
    x = MaxPooling2D(pool_size=config['pooling'])(x)

    # Layer-2: Conv2D-Conv2D-Pooling
    x = Conv2D(filters=config['filters'] // 2, kernel_size=config['kernel_2'], activation=config['activation'], padding=config['padding'])(x)
    x = Conv2D(filters=config['filters'] // 2, kernel_size=config['kernel_2'], activation=config['activation'], padding=config['padding'])(x)
    x = MaxPooling2D(pool_size=config['pooling'])(x)

    # Flatten and Fully Connected Head
    x = Flatten()(x)
    x = Dense(config['dense_units'], activation=config['activation'])(x)
    x = Dropout(config['dropout_f'])(x)

    out = Dense(num_classes, activation='softmax')(x)

    # Build and compile the model
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model