import cntk as c
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

c.cntk_py.set_fixed_random_seed(1)
c.cntk_py.force_deterministic_algorithms()
np.random.seed(0)
csv = './data.csv'
train_test_split_size = 0.2
minibatch_size = 16


def data_reading(file):
    data = pd.read_csv(file)
    useless_columns = ['Unnamed: 32', 'id', 'diagnosis', 'perimeter_mean', 'radius_mean', 'compactness_mean',
                       'concave points_mean', 'radius_se', 'perimeter_se',
                       'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst', 'compactness_se',
                       'concave points_se', 'texture_worst', 'area_worst']
    return data.drop(useless_columns, axis=1), data.diagnosis


def data_normalizing(x, y):
    x_norm = np.array((x - x.mean()) / (x.std()), dtype=np.float32)
    y_norm = []
    for i in y:
        new_label = ([1, 0]) if i == 'M' else ([0, 1])
        y_norm.append(new_label)
    return x_norm, np.array(y_norm, dtype=np.float32)


def model_init():
    return c.layers.Sequential([
        c.layers.Dense(60, init=c.glorot_uniform(), activation=c.ops.relu),
        c.layers.Dense(120, init=c.glorot_uniform(), activation=c.ops.relu),
        c.layers.Dropout(0.25),
        c.layers.Dense(240, init=c.glorot_uniform(), activation=c.ops.relu),
        c.layers.Dropout(0.25),
        c.layers.Dense(2, init=c.glorot_uniform(), activation=None)
    ])


def test_val():
    test_result = 0
    counter = 0
    for i in range(0, len(labels_test), minibatch_size):
        end = min(i + minibatch_size, len(labels_test))
        x = features_test[i:end]
        y = labels_test[i:end]
        eval_error = trainer.test_minibatch({input_var: x, label_var: y})
        test_result += eval_error
        counter += 1
    print("Average test error: {0:.5f}%".format(test_result / counter * 100))


def learn():
    start = time.time()
    for epoch in range(10):
        perm = np.random.permutation(len(labels_train))
        for i in range(0, len(labels_train), minibatch_size):
            end = min(i + minibatch_size, len(labels_train))
            x = features_train[perm[i:end]]
            y = labels_train[perm[i:end]]
            trainer.train_minibatch({input_var: x, label_var: y})
            progress_trace.update_with_trainer(trainer, with_metric=True)
        progress_trace.epoch_summary(with_metric=True)
        test_val()
    print("Time {:.1f} sec".format(time.time() - start))


if __name__ == '__main__':
    features, labels = data_reading(csv)
    features_norm, labels_norm = data_normalizing(features, labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features_norm,
                                                                                labels_norm,
                                                                                test_size=train_test_split_size)

    input_var = c.input_variable(16)
    label_var = c.input_variable(2)
    z = model_init()
    z = z(c.splice(input_var, c.ops.sqrt(input_var), c.ops.abs(input_var)))

    loss = c.cross_entropy_with_softmax(z, label_var)
    label_error = c.classification_error(z, label_var)
    c.logging.log_number_of_parameters(z)

    lr_per_minibatch = c.learning_rate_schedule(0.2, c.UnitType.minibatch)
    learner = c.sgd(z.parameters, lr_per_minibatch)
    progress_trace = c.logging.ProgressPrinter()

    trainer = c.Trainer(z, (loss, label_error), [learner], [progress_trace])
    learn()
