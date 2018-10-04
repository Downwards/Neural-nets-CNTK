import os
from cntk.io import *
from cntk.io.transforms import *
from cntk.layers import *
from cntk.ops import *
import numpy as np
import time
from PIL import Image

C.device.try_set_default_device(C.device.gpu(0))
source_dir = './CatDogs'
file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
train_image_folder = os.path.join(source_dir, 'train')
test_image_folder = os.path.join(source_dir, 'test')
minibatch_size = 64
num_epoch = 80
test_epoch_size = 2000
train_epoch_size = 20000
image_height = 128
image_width = 128


def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "map.txt")
    lines = []
    for class_id in range(0, len(class_mapping)):
        folder = os.path.join(root_folder, class_mapping[class_id])
        if os.path.exists(folder):
            for entry in os.listdir(folder):
                filename = os.path.join(folder, entry)
                if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                    lines.append("{0}\t{1}\n".format(filename, class_id))

    if include_unknown:
        for entry in os.listdir(root_folder):
            filename = os.path.join(root_folder, entry)
            if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                lines.append("{0}\t-1\n".format(filename))

    lines.sort()
    with open(map_file_name, 'w') as map_file:
        for line in lines:
            map_file.write(line)

    return map_file_name


def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    classes.sort()
    return np.asarray(classes)


def create_reader(map_file, train):
    trans = []
    if train:
        trans += [crop(crop_type='randomside', side_ratio=0.8)]
    trans += [scale(width=image_width, height=image_height, channels=3)]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=trans),
        labels=StreamDef(field='label', shape=2)
    )))


def create_model(model_input):
    return Sequential([
        For(range(3), lambda i: [
            Convolution((5, 5), [32, 32, 64][i], init=glorot_uniform(), pad=True, activation=relu),
            BatchNormalization(map_rank=1),
            MaxPooling((3, 3), strides=(2, 2))
        ]),
        Dense(256, init=glorot_uniform(), activation=relu),
        Dense(128, init=glorot_uniform(), activation=relu),
        Dropout(0.25),
        Dense(2, init=glorot_uniform(), activation=None)
    ])(model_input)


def test_val():
    counter = 0
    minibatch_index = 0
    metric_numer = 0
    while counter < test_epoch_size:
        current_minibatch = min(minibatch_size, test_epoch_size - counter)
        data = reader_test.next_minibatch(current_minibatch, input_map)
        metric_numer += trainer.test_minibatch(data)
        counter += data[label_var].num_samples
        minibatch_index += 1

    return (metric_numer * 100.0) / minibatch_index


def learn():
    start = time.time()
    for epoch in range(num_epoch):
        sample_count = 0
        while sample_count < train_epoch_size:
            data = reader_train.next_minibatch(min(minibatch_size, train_epoch_size - sample_count), input_map)
            trainer.train_minibatch(data)
            sample_count += data[label_var].num_samples
        progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)
        print("Eval test: {:0.1f}".format(test_val()))

    print("Time {:.1f} sec".format(time.time() - start))


def evaluate(image_path, trained_model):
    image_data = np.array(Image.open(image_path).resize((image_width, image_height)), dtype=np.float32)
    image_data = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))
    return trained_model.eval(image_data)


def evaluate_all():
    correct = 0
    total = 0
    with open(os.path.join(test_map_file)) as f:
        for x in f:
            real = int(x.split()[1])
            ev = evaluate(x.split()[0], z)
            res = 0 if ev[0][0] > ev[0][1] else 1
            if real == res:
                correct += 1
            total += 1

    print("Correct: {} of {}".format(correct, total))


if __name__ == '__main__':
    mapping = create_class_mapping_from_folder(train_image_folder)
    train_map_file = create_map_file_from_folder(train_image_folder, mapping)
    test_map_file = create_map_file_from_folder(test_image_folder, mapping, include_unknown=True)
    print("Map files created.")

    reader_train = create_reader(train_map_file, True)
    reader_test = create_reader(test_map_file, False)

    input_var = input_variable((3, image_height, image_width))
    label_var = input_variable(2)
    input_var_norm = element_times((1 / 256.0), minus(input_var, 128.0))
    z = create_model(input_var_norm)

    cross_er = cntk.cross_entropy_with_softmax(z, label_var)
    class_er = cntk.classification_error(z, label_var)
    lr_per_minibatch = cntk.learning_rate_schedule(0.0140, cntk.UnitType.minibatch)
    learner = cntk.adagrad(z.parameters, lr_per_minibatch)
    trainer = cntk.Trainer(z, (cross_er, class_er), [learner])

    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }
    progress_printer = cntk.logging.ProgressPrinter()

    cntk.logging.log_number_of_parameters(z)

    learn()
    evaluate_all()
