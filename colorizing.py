import cntk as c
from cntk.io import *
from cntk.layers import *
from cntk.ops import *
from PIL import Image
import numpy as np
import time
import os
import multiprocessing as mp
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray

c.cntk_py.set_fixed_random_seed(1)
c.cntk_py.force_deterministic_algorithms()
c.device.try_set_default_device(C.device.gpu(0))
np.random.seed(0)
image_size = 256
image_folder = r'C:\Users\Vladislav\Desktop\IT\NN\colorization\Train'
num_epoch = 1
minibatch_size = 8


def deserialize(image_path):
    image = np.array(Image.open(image_path).resize((image_size, image_size)))
    image_array = rgb2lab(1.0 / 255 * image)
    image_array = np.transpose(image_array, (2, 0, 1))

    x = image_array[0, :, ]
    x /= 100
    x = np.ascontiguousarray(x.reshape(1, 256, 256))

    y = image_array[1:, :, ]
    y /= 128
    y = np.ascontiguousarray(y)
    return x, y


def image_thread_process(tasks, features_p, labels_p):
    while True:
        new_value = tasks.get()
        if new_value == '_':
            break
        else:
            x, y = deserialize(new_value)
            features_p.put(x)
            labels_p.put(y)


def image_processing():
    manager = mp.Manager()
    images = manager.Queue()
    features_p = manager.Queue()
    labels_p = manager.Queue()

    for filename in os.listdir(image_folder):
        images.put(os.path.join(image_folder, filename))
    print('Total {} images.'.format(images.qsize()))

    num_proc = 12

    processes = []
    time_control = time.time()

    for i in range(num_proc):
        images.put('_')

    for i in range(num_proc):
        new_process = mp.Process(target=image_thread_process, args=(images, features_p, labels_p))
        processes.append(new_process)
        new_process.start()

    for proc in processes:
        proc.join()

    print('Deserialize - {:0.10f} seconds.'.format(time.time() - time_control))

    features_array = []
    labels_array = []

    for i in range(features_p.qsize()):
        features_array.append(features_p.get())

    for i in range(labels_p.qsize()):
        labels_array.append(labels_p.get())

    return np.asarray(features_array, dtype=np.float32), np.asarray(labels_array, dtype=np.float32)


def UpSampling2D(x):
    xr = c.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
    xx = c.splice(xr, xr, axis=-1)
    xy = c.splice(xx, xx, axis=-3)
    result = c.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))
    return result


def create_model(x):
    with C.layers.default_options(init=glorot_uniform()):
        m = Convolution((5, 5), 64, pad=True, activation=relu)(x)
        m = Convolution((3, 3), 64, strides=2, pad=True, activation=relu)(m)
        m = BatchNormalization()(m)
        m = Convolution((3, 3), 128, pad=True, activation=relu)(m)
        m = Convolution((3, 3), 128, strides=2, pad=True, activation=relu)(m)
        m = BatchNormalization()(m)
        m = Convolution((3, 3), 256, pad=True, activation=relu)(m)
        m = Convolution((3, 3), 256, strides=2, pad=True, activation=relu)(m)
        m = Convolution((3, 3), 512, pad=True, activation=relu)(m)
        m = Convolution((3, 3), 256, pad=True, activation=relu)(m)
        m = Convolution((3, 3), 128, pad=True, activation=relu)(m)
        m = UpSampling2D(m)
        m = Convolution((3, 3), 64, pad=True, activation=relu)(m)
        m = UpSampling2D(m)
        m = Convolution((3, 3), 32, pad=True, activation=relu)(m)
        m = UpSampling2D(m)
        m = Convolution((3, 3), 2, pad=True, activation=tanh)(m)
        return m


def mse(model, label):
    err = C.ops.reshape(C.ops.minus(model, label), model.shape)
    sq_err = C.ops.element_times(err, err)
    return C.ops.reduce_mean(sq_err)


def next_batch(feat, lab):
    for i in range(0, len(feat), minibatch_size):
        x = feat[i:i + minibatch_size]
        y = lab[i:i + minibatch_size]
        yield x, y


def learn():
    start = time.time()
    for epoch in range(num_epoch):
        counter = 0
        for f_batch, l_batch in next_batch(features, labels):
            trainer.train_minibatch({input_var: f_batch, label_var: l_batch})
            counter += minibatch_size
            if (counter % 512) == 0:
                progress_printer.update_with_trainer(trainer, with_metric=True)
                print("Finished Minibatch: loss = {:0.10f}, metric = {:0.10f}".format(
                    progress_printer.avg_loss_since_last(),
                    progress_printer.avg_metric_since_last()))
        trainer.summarize_training_progress()

    print("Learning time - {:.1f} sec".format(time.time() - start))


def colorize(path):
    x, y = deserialize(path)
    out = z.eval(x)
    out = out.reshape(2, 256, 256) * 128

    result = np.zeros((3, 256, 256))
    result[0, :, :] = x * 100
    result[1, :, :] = out[0, :, :]
    result[2, :, :] = out[1, :, :]
    result = np.transpose(result, (2, 0, 1))
    result = np.transpose(result, (2, 0, 1))
    imsave("img_result.png", lab2rgb(result))
    imsave("img_gray_version.png", rgb2gray(lab2rgb(result)))


if __name__ == '__main__':
    features, labels = image_processing()
    input_var = input_variable((1, image_size, image_size))
    label_var = input_variable((2, image_size, image_size))

    z = create_model(input_var)

    loss = mse(z, label_var)
    ev = mse(z, label_var)

    lr_rate = [0.001]
    lr_per_minibatch = c.learning_parameter_schedule(lr_rate, epoch_size=1)
    progress_printer = c.logging.ProgressPrinter()
    learner = c.adam(z.parameters, lr_per_minibatch, momentum=0.75)
    trainer = c.Trainer(z, (loss, ev), [learner], progress_printer)

    cntk.logging.log_number_of_parameters(z)

    learn()
    colorize('test.jpg')
