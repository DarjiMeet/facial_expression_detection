from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from expression_model import expression_model

import pickle
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth set.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)

def train_model():
    train_dir = "./facial_expression/train"
    test_dir = "./facial_expression/test"

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48,48),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    print("Class indices:", train_generator.class_indices)

    model = expression_model()
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=100,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        callbacks=[EarlyStopping(patience = 10 , restore_best_weights=True)]
    )

    return model ,history

if __name__ == "__main__":
    model, history = train_model()

    model.save('expression_model.keras')
    model.save_weights('expression_weights.h5')

    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)