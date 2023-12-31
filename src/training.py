from glob import glob
import math
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import RMSprop, Nadam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
import tensorflow as tf

languages = ['english', 'spanish', 'serbian']
categories = ['train', 'test']

data_root_path = '../data/'
train_path = data_root_path + 'train'

batch_size = 16
image_width = 500
image_height = 128

validation_split = 0.1
initial_learning_rate = 1e-3

num_classes = len(languages)

model_file = data_root_path + 'model.h5'


def step_decay(epoch, lr):
    drop = 0.94
    epochs_drop = 2.0
    lrate = lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def show_plot_training_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def limit_GPU_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == '__main__':
    limit_GPU_memory_growth()

    all_files = glob(train_path + '/*/*.png')

    num_validation = len(all_files) * validation_split
    num_train = len(all_files) - num_validation

    validation_steps = int(num_validation / batch_size)
    steps_per_epoch = int(num_train / batch_size)

    print('Steps per Epoch: ' + str(steps_per_epoch))
    print('Validation steps: ' + str(validation_steps))

    tensor_board_filename = 'inceptionV3_board'
    tensorboard = TensorBoard(log_dir='logs/{}'.format(tensor_board_filename))

    image_data_generator = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    train_generator = image_data_generator.flow_from_directory(train_path, batch_size=batch_size, class_mode='categorical', target_size=(image_height, image_width), color_mode='grayscale', subset='training')
    validation_generator = image_data_generator.flow_from_directory(train_path, batch_size=batch_size, class_mode='categorical', target_size=(image_height, image_width), color_mode='grayscale', subset='validation')

    img_input = Input(shape=(image_height, image_width, 1))

    img_conc = Concatenate(axis=3, name='input_concat')([img_input, img_input, img_input])

    model = InceptionV3(input_tensor=img_conc, weights=None, include_top=True, classes=num_classes)

    # model.summary()

    model.compile(optimizer=Nadam(learning_rate=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)
    learning_rate_decay = LearningRateScheduler(step_decay, verbose=1)

    history = model.fit(train_generator, validation_data=validation_generator, epochs=60, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[early_stopping, learning_rate_decay, tensorboard])
    model.save(model_file)

    show_plot_training_history(history)
    
