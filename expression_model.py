from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Multiply, LeakyReLU, add, Activation, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def expression_model(input_shape=(48,48,1), num_classes=7):
    inputs = Input(shape=input_shape)

    # Entry Flow
    x = Conv2D(32, 3, strides=2, padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(64, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Middle Flow
    for _ in range(4):
        residual = x
        x = Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(128, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=1, padding='same')(x)

        if residual.shape[-1] != x.shape[-1]:
            residual = Conv2D(128, (1, 1), strides=1, padding='same')(residual)

        x = add([x, residual])

    # Attention
    attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = Multiply()([x, attention])

    # Exit Flow
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, output)

model = expression_model()
model.summary()
