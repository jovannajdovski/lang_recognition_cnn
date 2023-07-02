from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob


data_root_path = '../data/'
model_file = data_root_path + 'model.h5'
test_path = data_root_path + 'test'

batch_size = 32
image_width = 500
image_height = 128

if __name__ == '__main__':
    all_files = glob(test_path + '/*/*.png')
    num_test = len(all_files)
    evaluation_steps = int(num_test / batch_size)
    print('Evaluation steps: ' + str(evaluation_steps))

    image_data_generator = ImageDataGenerator(rescale=1./255)
    evaluation_generator = image_data_generator.flow_from_directory(test_path, batch_size=batch_size, class_mode='categorical', target_size=(image_height, image_width), color_mode='grayscale')

    model = load_model(model_file)
    _, test_accuracy = model.evaluate(evaluation_generator, steps=evaluation_steps)

    print('Test accuracy: ' + str(round(test_accuracy * 100, 1)) + ' %')