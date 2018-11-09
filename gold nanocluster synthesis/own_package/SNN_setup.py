import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Convolution2D, Flatten, merge, Input, Lambda, Concatenate
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
import pickle, time, collections, gc

# Own Scripts
from .features_labels_setup import Features_labels, read_reaction_data_smiles

# NGF
from .NGF.models import build_graph_conv_net_fp_only
from .NGF.layers import NeuralGraphHidden, NeuralGraphOutput


def create_SNN_hparams(hidden_layers=[30], learning_rate=0.001, epochs=100, batch_size=64, pair_size=32,
                       activation='relu',
                       optimizer='Adam', singlenet_l1=0, singlenet_l2=0, reg_term=0, feature_vector_dim=10, dropout=0,
                       fp_length=100, fp_number=3, conv_width=8, conv_number=2,
                       conv_activation='relu', conv_l1=0, conv_l2=0,
                       verbose=1):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['hidden_layers', 'learning_rate', 'epochs', 'batch_size', 'pair_size', 'activation', 'optimizer',
             'singlenet_l1', 'singlenet_l2',
             'reg_term', 'dropout', 'feature_vector_dim', 'fp_length', 'fp_number', 'conv_width', 'conv_number',
             'conv_activation', 'conv_l1', 'conv_l2',
             'verbose']
    values = [hidden_layers, learning_rate, epochs, batch_size, pair_size, activation, optimizer,
              singlenet_l1, singlenet_l2,
              reg_term, dropout, feature_vector_dim, fp_length, fp_number, conv_width, conv_number,
              conv_activation, conv_l1, conv_l2,
              verbose]
    hparams = dict(zip(names, values))
    return hparams


class SNN:
    def __init__(self, SNN_hparams, fl):
        self.hparams = SNN_hparams
        self.features_c_dim = fl.features_c_dim

        # Second step: Concat features_c and fingerprint vectors ==> form intermediate input vector (IIV)
        # IIV put into single SNN net
        # Defining half_model_model
        half_c = Input(shape=(fl.features_c_dim,))

        # singlenet for the SNN. Same weights and biases for top and bottom half of SNN
        singlenet = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        numel = len(hidden_layers)
        generator_dropout = self.hparams.get('dropout', 0)
        singlenet.add(Dense(hidden_layers[0],
                            input_dim=self.features_c_dim,
                            activation=self.hparams['activation'],
                            kernel_regularizer=regularizers.L1L2(l1=self.hparams['singlenet_l1'],
                                                                 l2=self.hparams['singlenet_l2'])))
        if generator_dropout != 0:
            singlenet.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    singlenet.add(Dense(hidden_layers[i + 1],
                                        activation=self.hparams['activation'],
                                        kernel_regularizer=regularizers.L1L2(l1=self.hparams['singlenet_l1'],
                                                                             l2=self.hparams['singlenet_l2'])))
        singlenet.add(Dense(self.hparams.get('feature_vector_dim', 10), activation='sigmoid'))

        # Output of half_model
        encoded_half = singlenet(half_c)

        # Make half_model callable
        half_model = Model(name='half_model_encoded',
                           input=half_c,
                           output=encoded_half)

        # All steps together by calling the above models one after another.
        # fp model ==> half_model ==> L1 distance and final node model
        lc = Input(name='left_c', shape=(fl.features_c_dim,))
        rc = Input(name='right_c', shape=(fl.features_c_dim,))

        # Apply half_model to left and right side
        encoded_l = half_model(lc)
        encoded_r = half_model(rc)

        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)
        self.siamese_net = Model(name='final_model',
                                 input=[lc,rc],
                                 output=prediction)

        if self.hparams.get('learning_rate', None) is None:
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=self.hparams['optimizer'])
        else:
            sgd = optimizers.Adam(lr=self.hparams['learning_rate'])
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=sgd)

class SNN_backup:
    def __init__(self, SNN_hparams, fl):
        self.hparams = SNN_hparams
        self.features_c_dim = fl.features_c_dim

        left_input = Input(shape=(self.features_c_dim,))
        right_input = Input(shape=(self.features_c_dim,))

        encoded_l = self.singlenet()(left_input)
        encoded_r = self.singlenet()(right_input)

        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)
        self.siamese_net = Model(input=[left_input, right_input], output=prediction)

        if self.hparams.get('learning_rate', None) is None:
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=self.hparams['optimizer'])
        else:
            sgd = optimizers.Adam(lr=self.hparams['learning_rate'])
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=sgd)

    def singlenet(self):
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        numel = len(hidden_layers)
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_c_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.hparams.get('feature_vector_dim', 10), activation='sigmoid'))
        return model


class Siamese_loader:
    def __init__(self, model, x_train, SNN_hparams):
        """
        :param model: SNN class model
        :param x_train: Features_labels class containing training dataset.
        """
        # General attributes
        if isinstance(model, list):
            # If model is given as a list, means using eval_ensemble ONLY!
            self.model_store = model
        else:
            self.model = model
        self.x_train = x_train
        self.n_classes = x_train.n_classes
        self.n_examples = x_train.count
        self.features_c_norm = x_train.features_c_norm
        self.labels = x_train.labels
        self.features_c_dim = self.features_c_norm[0][1].shape[1]
        self.hparams = SNN_hparams

    def get_batch(self, batch_size):
        categories = rng.choice(self.n_classes, size=(batch_size,), replace=True)
        pairs = [np.zeros((batch_size, self.features_c_dim)) for _ in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]  # Set the current category (aka class) for the loop
            n_examples = self.features_c_norm[category][1].shape[0]  # Get no. of examples for current class in the loop
            idx_1 = rng.randint(0, n_examples)  # Choose one example out of the examples in the class
            # [left_c, left_d, right_c, right_d]
            pairs[0][i, :] = self.features_c_norm[category][1][idx_1, :]  # Class, features, example, all input features
            if i >= batch_size // 2:
                category_2 = category
            else:
                category_2 = (category + rng.randint(1, self.n_classes)) % self.n_classes  # Randomly choose diff class
                n_examples = self.features_c_norm[category_2][1].shape[0]
            idx_2 = rng.randint(0, n_examples)
            pairs[1][i, :] = self.features_c_norm[category_2][1][idx_2, :]
        return pairs, targets

    def get_batch1(self, batch_size):
        categories = rng.choice(self.n_classes, size=(batch_size,), replace=True)
        pairs = [np.zeros((batch_size, self.features_c_dim)) for _ in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]  # Set the current category (aka class) for the loop
            n_examples = self.features_c_norm[category][1].shape[0]  # Get no. of examples for current class in the loop
            idx_1 = rng.randint(0, n_examples)  # Choose one example out of the examples in the class
            # [left_c, left_d, right_c, right_d]
            pairs[0][i, :] = self.features_c_norm[category][1][idx_1, :]  # Class, features, example, all input features
            if i >= batch_size // 2:
                category_2 = category
            else:
                category_2 = (category + rng.randint(1, self.n_classes)) % self.n_classes  # Randomly choose diff class
                n_examples = self.features_c_norm[category_2][1].shape[0]
            idx_2 = rng.randint(0, n_examples)
            pairs[1][i, :] = self.features_c_norm[category_2][1][idx_2, :]
        return pairs, targets

    def get_batch2(self, batch_size):
        time1start = time.time()
        categories = rng.choice(self.n_classes, size=(batch_size,), replace=True)
        pairs = [[] for _ in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]  # Set the current category (aka class) for the loop
            n_examples = self.features_c_norm[category][1].shape[0]  # Get no. of examples for current class in the loop
            idx_1 = rng.randint(0, n_examples)  # Choose one example out of the examples in the class
            # [left_c, left_d, right_c, right_d]
            pairs[0].append(
                self.features_c_norm[category][1][idx_1, ...])  # Class, features, example, all input features
            if i >= batch_size // 2:
                category_2 = category
            else:
                category_2 = (category + rng.randint(1, self.n_classes)) % self.n_classes  # Randomly choose diff class
                n_examples = self.features_c_norm[category_2][1].shape[0]
            idx_2 = rng.randint(0, n_examples)
            pairs[1].append(self.features_c_norm[category_2][1][idx_2, ...])
        time1end = time.time()
        pairs = [np.array(pair) for pair in pairs]
        print('Time taken: {0}'.format(time1end - time1start))
        return pairs, targets

    def generate(self, batch_size):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size)
            yield (pairs, targets)

    def train(self, steps, batch_size, verbose=1, save_mode=False, save_dir='./save/models/', save_name='SNN.h5'):
        self.model.fit_generator(generator=self.generate(batch_size), steps_per_epoch=batch_size, epochs=steps,
                                 verbose=verbose)
        if save_mode:
            self.model.save(save_dir + save_name)
        return self.model

    def get_oneshot_predict_batch(self, category, x_predict_c_norm):
        x_category_c_norm = self.features_c_norm[category][1]
        n_examples_category = x_category_c_norm.shape[0]
        pairs = []
        pairs.append(x_category_c_norm)
        pairs.append(np.repeat(x_predict_c_norm, n_examples_category, axis=0))
        return pairs

    def oneshot_predict(self, x_predict_c_norm_a, compare_mode=False, print_mode=False):
        """

        :param x_predict: np array of no. examples x features_dim
        :param compare_mode: compare between predicted and actual labels
        :return: Confusion matrix
        """
        n_examples_predict = x_predict_c_norm_a.shape[0]
        model = self.model
        predicted_class_store = []
        for j in range(n_examples_predict):  # Looping through all examples
            # Store the final SNN node output when comparing current example with support set examples
            predicted_labels_store = []
            for i in range(self.n_classes):  # For one example, check through all classes
                pairs = self.get_oneshot_predict_batch(category=i,
                                                       x_predict_c_norm=x_predict_c_norm_a[np.newaxis, j, :])
                # Vector of scores for one example against one class
                predicted_labels = model.predict(pairs)
                # Avg score for that one class
                n_class_example_count = self.features_c_norm[i][1].shape[0]
                predicted_labels_store.append(np.sum(predicted_labels) / n_class_example_count)
            # After checking through all classes, select the class with the highest avg score and store it
            predicted_class_store.append(predicted_labels_store.index(max(predicted_labels_store)))
        return predicted_class_store

    def eval(self, eval_fl):
        eval_start = time.time()
        x_predict_c_norm_a = eval_fl.features_c_norm_a
        labels_hot = eval_fl.labels_hot
        labels = eval_fl.labels
        n_examples_predict = x_predict_c_norm_a.shape[0]
        model = self.model
        predicted_class_store = []
        predicted_labels_hot_store = []
        for j in range(n_examples_predict):  # Looping through all examples
            # Store the final SNN node output when comparing current example with support set examples
            predicted_labels_hot = []
            for i in range(self.n_classes):  # For one example, check through all classes
                pairs = self.get_oneshot_predict_batch(category=i,
                                                       x_predict_c_norm=x_predict_c_norm_a[np.newaxis, j, :])
                # Vector of scores for one example against one class
                predicted_labels = model.predict(pairs)
                # Avg score for that one class
                n_class_example_count = self.features_c_norm[i][1].shape[0]
                predicted_labels_hot.append(np.sum(predicted_labels) / n_class_example_count)
            # After checking through all classes, select the class with the highest avg score and store it
            predicted_class_store.append(predicted_labels_hot.index(max(predicted_labels_hot)))
            predicted_labels_hot_store.append(predicted_labels_hot)
        predicted_labels_hot_store = np.array(predicted_labels_hot_store)
        acc = accuracy_score(labels, predicted_class_store)
        ce = log_loss(labels_hot, predicted_labels_hot_store)
        cm = confusion_matrix(labels, predicted_class_store)
        try:
            f1s = f1_score(labels, predicted_class_store)  # Will work for binary classification
        except ValueError:  # Multi-class will raise ValueError from sklearn f1_score function
            f1s = f1_score(labels, predicted_class_store, average='micro')  # Must use micro averaging for multi-class
        mcc = matthews_corrcoef(labels, predicted_class_store)
        eval_end = time.time()
        print('eval run time : {}'.format(eval_end - eval_start))
        return predicted_class_store, acc, ce, cm, f1s, mcc

    def eval_ensemble(self, eval_fl):
        n_examples_predict = eval_fl.count
        total_model_count = len(self.model_store)
        ensemble_predicted_class_store = []
        ensemble_labels_hot_store = []
        for instance, model_path in enumerate(self.model_store):
            # Stuff to ensure keras can run smoothly due to training multiple model instances in the same script
            sess = tf.Session()
            K.set_session(sess)

            eval_start = time.time()
            model = load_model(model_path)
            predicted_class_store = []
            predicted_labels_hot_store = []
            for j in range(n_examples_predict):  # Looping through all examples
                # Store the final SNN node output when comparing current example with support set examples
                predicted_labels_hot = []
                for i in range(self.n_classes):  # For one example, check through all classes
                    pairs = self.get_oneshot_predict_batch(category=i,
                                                           x_predict_c_norm=eval_fl.features_c_norm_a[np.newaxis, j, :])
                    # Vector of scores for one example against one class
                    predicted_labels = model.predict(pairs)
                    # Avg score for that one class
                    n_class_example_count = self.features_c_norm[i][1].shape[0]
                    predicted_labels_hot.append(np.sum(predicted_labels) / n_class_example_count)
                # After looping through all the support classes, store the label hot for that particular eval example
                predicted_labels_hot_store.append(predicted_labels_hot)
                # Select the class with the highest avg score and store it
                predicted_class_store.append(predicted_labels_hot.index(max(predicted_labels_hot)))
            predicted_labels_hot_store = np.array(predicted_labels_hot_store)  # shape (n_examples, n_classes)
            if ensemble_labels_hot_store != []:
                ensemble_labels_hot_store += predicted_labels_hot_store
            else:
                ensemble_labels_hot_store = predicted_labels_hot_store
            ensemble_predicted_class_store.append(predicted_class_store)  # shape (No. of models, n_examples)

            # Stuff to ensure keras can run smoothly due to training multiple model instances in the same script
            del model
            K.clear_session()
            gc.collect()
            eval_end = time.time()

            print('{} out of {}: {} run time  {}'.format(instance + 1, total_model_count, model_path,
                                                         eval_end - eval_start))

        final_ensemble_predicted_class = np.argmax(ensemble_labels_hot_store, axis=1)
        ensemble_labels_hot_store = np.array(ensemble_labels_hot_store) / total_model_count
        return ensemble_predicted_class_store, final_ensemble_predicted_class, ensemble_labels_hot_store



class SNN_smiles:
    def __init__(self, SNN_hparams, fl):
        self.hparams = SNN_hparams
        self.features_c_dim = fl.features_c_dim
        self.features_d_count = fl.features_d_count
        self.fp_length = [SNN_hparams['fp_length'] for _ in range(SNN_hparams['fp_number'])]
        self.conv_width = [SNN_hparams['conv_width'] for _ in range(SNN_hparams['conv_number'])]

        # First step: SMILES ==> fingerprint vector
        # Defining fingerprint(fp) model
        half_features_d = []
        half_conv_net = []
        for idx in range(fl.features_d_count):
            # Creating left input tensors for features_d.
            # For each molecule
            # Make one input tensor for atoms, bond, edge tensor
            half_features_d.append([Input(name='h_a_inputs_' + str(idx) + 'x', shape=fl.features_d_a[idx][0].shape[1:]),
                                    Input(name='h_b_inputs_' + str(idx) + 'y', shape=fl.features_d_a[idx][1].shape[1:]),
                                    Input(name='h_e_inputs_' + str(idx) + 'z', shape=fl.features_d_a[idx][2].shape[1:],
                                          dtype='int32')]
                                   )
            single_molecule = half_features_d[-1]
            # Building the half_conv_net for that particular molecule
            single_molecule_half_conv_net = build_graph_conv_net_fp_only(single_molecule,
                                                                         conv_layer_sizes=self.conv_width,
                                                                         fp_layer_size=self.fp_length,
                                                                         conv_activation=self.hparams[
                                                                             'conv_activation'],
                                                                         conv_l1=self.hparams['conv_l1'],
                                                                         conv_l2=self.hparams['conv_l2'],
                                                                         fp_activation='softmax')
            single_molecule_half_conv_net_model = Model(name='h_fp_' + str(idx),
                                                        inputs=single_molecule,
                                                        outputs=single_molecule_half_conv_net)
            half_conv_net.append(single_molecule_half_conv_net_model)

        # Second step: Concat features_c and fingerprint vectors ==> form intermediate input vector (IIV)
        # IIV put into single SNN net
        # Defining half_model_model
        half_c = Input(shape=(fl.features_c_dim,))
        half_fp = [Input(shape=(self.fp_length[0],)) for _ in range(fl.features_d_count)]

        # Concat left side 4 inputs and right side 4 inputs
        half_combined = merge.Concatenate()([half_c] + half_fp)

        # singlenet for the SNN. Same weights and biases for top and bottom half of SNN
        singlenet = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        numel = len(hidden_layers)
        generator_dropout = self.hparams.get('dropout', 0)
        singlenet.add(Dense(hidden_layers[0],
                            input_dim=self.features_c_dim + self.features_d_count * self.hparams['fp_length'],
                            activation=self.hparams['activation'],
                            kernel_regularizer=regularizers.L1L2(l1=self.hparams['singlenet_l1'],
                                                                 l2=self.hparams['singlenet_l2'])))
        if generator_dropout != 0:
            singlenet.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    singlenet.add(Dense(hidden_layers[i + 1],
                                        activation=self.hparams['activation'],
                                        kernel_regularizer=regularizers.L1L2(l1=self.hparams['singlenet_l1'],
                                                                             l2=self.hparams['singlenet_l2'])))
        singlenet.add(Dense(self.hparams.get('feature_vector_dim', 10), activation='sigmoid'))

        # Output of half_model
        encoded_half = singlenet(half_combined)

        # Make half_model callable
        half_model = Model(name='half_model_encoded',
                           input=[half_c] + half_fp,
                           output=encoded_half)

        # All steps together by calling the above models one after another.
        # fp model ==> half_model ==> L1 distance and final node model
        lc = Input(name='left_c', shape=(fl.features_c_dim,))
        left_features_d = []
        left_fp_model = []
        rc = Input(name='right_c', shape=(fl.features_c_dim,))
        right_features_d = []
        right_fp_model = []
        for idx in range(fl.features_d_count):
            # a = atom tensor, b = bond tensor, e = edge tensor
            left_features_d.append([Input(name='l_a_inputs_' + str(idx) + 'x', shape=fl.features_d_a[idx][0].shape[1:]),
                                    Input(name='l_b_inputs_' + str(idx) + 'y', shape=fl.features_d_a[idx][1].shape[1:]),
                                    Input(name='l_e_inputs_' + str(idx) + 'z', shape=fl.features_d_a[idx][2].shape[1:],
                                          dtype='int32')]
                                   )
            # Call fp model for each set of left molecules
            left_fp_model.append(half_conv_net[idx](left_features_d[-1]))
            # Same as left side. Just change left to right.
            right_features_d.append(
                [Input(name='r_a_inputs_' + str(idx) + 'x', shape=fl.features_d_a[idx][0].shape[1:]),
                 Input(name='r_b_inputs_' + str(idx) + 'y', shape=fl.features_d_a[idx][1].shape[1:]),
                 Input(name='r_e_inputs_' + str(idx) + 'z', shape=fl.features_d_a[idx][2].shape[1:],
                       dtype='int32')]
            )
            # Call fp model for each set of right molecules
            right_fp_model.append(half_conv_net[idx](right_features_d[-1]))

        # Apply half_model to left and right side
        encoded_l = half_model([lc] + left_fp_model)
        encoded_r = half_model([rc] + right_fp_model)

        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)
        self.siamese_net = Model(name='final_model',
                                 input=[lc] + [left_tensor for molecule in left_features_d for left_tensor in
                                               molecule] +
                                       [rc] + [right_tensor for molecule in right_features_d for right_tensor in
                                               molecule],
                                 output=prediction)

        if self.hparams.get('learning_rate', None) is None:
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=self.hparams['optimizer'])
        else:
            sgd = optimizers.Adam(lr=self.hparams['learning_rate'])
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=sgd)


class SNN_smiles_backup:
    def __init__(self, SNN_hparams, fl):
        self.hparams = SNN_hparams
        self.features_c_dim = fl.features_c_dim
        self.features_d_count = fl.features_d_count
        self.fp_length = [SNN_hparams['fp_length'] for _ in range(SNN_hparams['fp_number'])]
        self.conv_width = [SNN_hparams['conv_width'] for _ in range(SNN_hparams['conv_number'])]
        # Left side
        lc = Input(shape=(fl.features_c_dim,))
        left_features_d = []
        left_conv_net = []
        for idx in range(fl.features_d_count):
            # Creating left input tensors for features_d.
            # For each molecule
            # Make one input tensor for atoms, bond, edge tensor
            left_features_d.append([Input(name='l_a_inputs_' + str(idx) + 'x', shape=fl.features_d_a[idx][0].shape[1:]),
                                    Input(name='l_b_inputs_' + str(idx) + 'y', shape=fl.features_d_a[idx][1].shape[1:]),
                                    Input(name='l_e_inputs_' + str(idx) + 'z', shape=fl.features_d_a[idx][2].shape[1:],
                                          dtype='int32')]
                                   )
            # Building the left_conv_net for that particular molecule
            left_conv_net.append(build_graph_conv_net_fp_only(left_features_d[idx],
                                                              conv_layer_sizes=self.conv_width,
                                                              fp_layer_size=self.fp_length,
                                                              conv_activation='relu', fp_activation='softmax'))

        # Right side
        rc = Input(shape=(fl.features_c_dim,))
        right_features_d = []
        right_conv_net = []
        for idx in range(fl.features_d_count):
            # Same as left side. Just change left to right.
            right_features_d.append(
                [Input(name='r_a_inputs_' + str(idx) + 'x', shape=fl.features_d_a[idx][0].shape[1:]),
                 Input(name='r_b_inputs_' + str(idx) + 'y', shape=fl.features_d_a[idx][1].shape[1:]),
                 Input(name='r_e_inputs_' + str(idx) + 'z', shape=fl.features_d_a[idx][2].shape[1:],
                       dtype='int32')]
            )
            right_conv_net.append(build_graph_conv_net_fp_only(right_features_d[idx],
                                                               conv_layer_sizes=self.conv_width,
                                                               fp_layer_size=self.fp_length,
                                                               conv_activation='relu', fp_activation='softmax')
                                  )

        # Concat left side 4 inputs and right side 4 inputs
        left_combined = merge.Concatenate()([lc] + left_conv_net)
        right_combined = merge.Concatenate()([rc] + right_conv_net)

        encoded_l = self.singlenet()(left_combined)
        encoded_r = self.singlenet()(right_combined)

        # layer to merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # call this layer on list of two input tensors.
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)
        self.siamese_net = Model(
            input=[lc] + [left_tensor for molecule in left_features_d for left_tensor in molecule] +
                  [rc] + [right_tensor for molecule in right_features_d for right_tensor in molecule],
            output=prediction)

        if self.hparams.get('learning_rate', None) is None:
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=self.hparams['optimizer'])
        else:
            sgd = optimizers.Adam(lr=self.hparams['learning_rate'])
            self.siamese_net.compile(loss='binary_crossentropy', optimizer=sgd)

    def singlenet(self):
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        numel = len(hidden_layers)
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_c_dim + self.features_d_count * self.hparams['fp_length'],
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.hparams.get('feature_vector_dim', 10), activation='sigmoid'))
        return model


class Siamese_loader_smiles:
    def __init__(self, model, x_train, SNN_hparams=None):
        """
        :param model: SNN class model
        :param x_train: Features_labels class containing training dataset.
        """
        if SNN_hparams is None:
            assert isinstance(model, list), 'SNN_hparams is only None if model given is an ensemble of model and only' \
                                            'eval_ensemble is going to be used.'

        # General attributes
        if isinstance(model, list):
            # If model is given as a list, means using eval_ensemble ONLY!
            self.model_store = model
        else:
            self.model = model
        self.hparams = SNN_hparams
        self.x_train = x_train
        self.n_classes = x_train.n_classes
        self.n_examples = x_train.count
        # features_c and d
        self.features_c_norm = x_train.features_c_norm
        self.features_c_dim = self.features_c_norm[0][1].shape[1]
        self.features_d = x_train.features_d
        # labels
        self.labels = x_train.labels

    def get_batch(self, batch_size):
        categories = rng.choice(self.n_classes, size=(batch_size,), replace=True)
        # Create list of empty list. Each side of the SNN gets 1 + 3*features_d_count inputs. Times 2 for left and right
        pairs = [[] for _ in range((1 + self.x_train.features_d_count * 3) * 2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            # Set the left category (aka class) for the loop
            category = categories[i]
            # Get no. of examples for left class in the loop
            n_examples = self.features_c_norm[category][1].shape[0]
            # Choose one random example out of the examples in the left class
            idx_1 = rng.randint(0, n_examples)
            # Append features c to the 0th list in the pairs nested list
            idx = 0
            pairs[idx].append(self.features_c_norm[category][1][idx_1, ...])
            # Add one to the counter idx that is keeping track of the current position in the nested list
            idx += 1
            # Adding information about features_d
            for single_molecule in self.features_d:
                for single_tensor in single_molecule:
                    pairs[idx].append(single_tensor[category][1][idx_1, ...])
                    idx += 1
            # Setting right side
            # Half same class, half different class
            if i >= batch_size // 2:
                # Same class examples for the later half of the batch
                category_2 = category
            else:
                # Different class examples for the first half of the batch
                category_2 = (category + rng.randint(1, self.n_classes)) % self.n_classes  # Randomly choose diff class
                # Since different class will have different number of examples in it to randomly select from
                # need to set new value to n_examples to randomly select examples from the different class.
                n_examples = self.features_c_norm[category_2][1].shape[0]
            # Choose one random example from right class
            idx_2 = rng.randint(0, n_examples)
            # Next few lines are same as left side. Just change category 1 to 2, idx 1 to 2
            pairs[idx].append(self.features_c_norm[category_2][1][idx_2, ...])
            idx += 1
            for single_molecule in self.features_d:
                for single_tensor in single_molecule:
                    pairs[idx].append(single_tensor[category_2][1][idx_2, ...])
                    idx += 1
        # Convert nested list to a list of ndarray
        pairs = [np.array(pair) for pair in pairs]
        # print([pair.shape for pair in pairs])
        return pairs, targets

    def generate(self, batch_size):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size)
            yield (pairs, targets)

    def train(self, steps, batch_size, pair_size, verbose=1, save_mode=False, save_dir='./save/models/',
              save_name='SNN.h5'):
        '''
        Method to call to train keras SNN_smiles model
        :param steps: Number of steps to train model on. 1 step is training the model for 1 iteration on 1 batch
        :param batch_size: Number of set of pairs per epoch
        :param pair_size: Number of pairs of examples in 1 batch
        :param verbose: Keras verbose. 0 = silent, 1 = full details, 2 = less details
        :param save_mode: to determine whether or not to save the model
        :param save_dir: directory the model will be saved into
        :param save_name: name of the saved model
        :return:
        '''
        self.model.fit_generator(generator=self.generate(pair_size), steps_per_epoch=batch_size, epochs=steps,
                                 verbose=verbose)
        if save_mode:
            # keras saved model must be saved with file extension .h5 at the end
            if save_name[-3] != '.h5':
                save_name = save_name + '.h5'
            self.model.save(save_dir + save_name)
        return self.model

    def get_oneshot_predict_batch(self, category, eval_idx, eval_fl):
        n_examples_category = self.features_c_norm[category][1].shape[0]
        pairs = [[] for _ in range((1 + self.x_train.features_d_count * 3) * 2)]
        # Left side inputs. Left input rows represents all the examples for that particular category in the support set
        # Append features c to the 0th list in the pairs nested list. Take entire example in that support set.
        idx = 0
        pairs[idx] = self.features_c_norm[category][1]
        # Add one to the counter idx that is keeping track of the current position in the nested list
        idx += 1
        # Adding information about left features_d for the support set
        for single_molecule in self.features_d:
            for single_tensor in single_molecule:
                pairs[idx] = single_tensor[category][1]
                idx += 1

        # Right side inputs. Right side rows is all the same, containing the same eval example that is being evaluated.
        # Append features c to the current idx list in the pairs nested list
        pairs[idx] = np.repeat(eval_fl.features_c_norm_a[None, eval_idx, ...], n_examples_category, axis=0)
        # Add one to the counter idx that is keeping track of the current position in the nested list
        idx += 1
        # Adding information about features_d for eval example
        for single_molecule in eval_fl.features_d_a:
            for single_tensor in single_molecule:
                # Since there is only 1 eval example but many examples in the support set
                # dublicate the eval example n_examples_category number of times to match support set shape.
                pairs[idx] = np.repeat(single_tensor[None, eval_idx, ...], n_examples_category, axis=0)
                idx += 1
        return pairs

    def eval(self, eval_fl):
        eval_start = time.time()
        # One hot encoding for labels
        labels_hot = eval_fl.labels_hot
        # Dense representation for labels
        labels = eval_fl.labels
        # Total number of examples to evaluate
        n_examples_predict = eval_fl.count
        model = self.model
        # Store the predicted class for the eval examples. [class_1, class_2, class_3, ... , class_last_eval]
        predicted_class_store = []
        # Store predicted hot labels for the eval examples. Nested list (no. of eval examples, total no. of classes)
        predicted_labels_hot_store = []
        for j in range(n_examples_predict):  # Looping through all examples
            # Store the final SNN node output when comparing current example with support set examples
            predicted_labels_hot = []
            for i in range(self.n_classes):  # For one example, check through all classes
                pairs = self.get_oneshot_predict_batch(category=i, eval_idx=j, eval_fl=eval_fl)
                # Vector of scores for one example against one class. shape = (no. of support set,)
                predicted_labels = model.predict(pairs)
                # Avg score for that one support class.
                n_class_example_count = self.features_c_norm[i][1].shape[0]
                predicted_labels_hot.append(np.sum(predicted_labels) / n_class_example_count)
            # After looping through all the support classes, store the label hot for that particular eval example
            predicted_labels_hot_store.append(predicted_labels_hot)
            # Select the class with the highest avg score and store it
            predicted_class_store.append(predicted_labels_hot.index(max(predicted_labels_hot)))

        # Convert nest list to ndarray
        predicted_labels_hot_store = np.array(predicted_labels_hot_store)

        # Compute some evaluation metrics
        acc = accuracy_score(labels, predicted_class_store)
        ce = log_loss(labels_hot, predicted_labels_hot_store)
        cm = confusion_matrix(labels, predicted_class_store)
        try:
            f1s = f1_score(labels, predicted_class_store)  # Will work for binary classification
        except ValueError:  # Multi-class will raise ValueError from sklearn f1_score function
            f1s = f1_score(labels, predicted_class_store, average='micro')  # Must use micro averaging for multi-class
        mcc = matthews_corrcoef(labels, predicted_class_store)
        eval_end = time.time()
        print('eval run time : {}'.format(eval_end - eval_start))
        return predicted_class_store, acc, ce, cm, f1s, mcc

    def eval_ensemble(self, eval_fl):
        n_examples_predict = eval_fl.count
        total_model_count = len(self.model_store)
        ensemble_predicted_class_store = []
        ensemble_labels_hot_store = []
        for instance, model_path in enumerate(self.model_store):
            # Stuff to ensure keras can run smoothly due to training multiple model instances in the same script
            sess = tf.Session()
            K.set_session(sess)

            eval_start = time.time()
            model = load_model(model_path, custom_objects={'NeuralGraphHidden': NeuralGraphHidden,
                                                           'NeuralGraphOutput': NeuralGraphOutput})
            predicted_class_store = []
            predicted_labels_hot_store = []
            for j in range(n_examples_predict):  # Looping through all examples
                # Store the final SNN node output when comparing current example with support set examples
                predicted_labels_hot = []
                for i in range(self.n_classes):  # For one example, check through all classes
                    pairs = self.get_oneshot_predict_batch(category=i, eval_idx=j, eval_fl=eval_fl)
                    # Vector of scores for one example against one class
                    predicted_labels = model.predict(pairs)
                    # Avg score for that one class
                    n_class_example_count = self.features_c_norm[i][1].shape[0]
                    predicted_labels_hot.append(np.sum(predicted_labels) / n_class_example_count)
                # After looping through all the support classes, store the label hot for that particular eval example
                predicted_labels_hot_store.append(predicted_labels_hot)
                # Select the class with the highest avg score and store it
                predicted_class_store.append(predicted_labels_hot.index(max(predicted_labels_hot)))
            predicted_labels_hot_store = np.array(predicted_labels_hot_store)  # shape (n_examples, n_classes)
            if ensemble_labels_hot_store != []:
                ensemble_labels_hot_store += predicted_labels_hot_store
            else:
                ensemble_labels_hot_store = predicted_labels_hot_store
            ensemble_predicted_class_store.append(predicted_class_store)  # shape (No. of models, n_examples)

            # Stuff to ensure keras can run smoothly due to training multiple model instances in the same script
            del model
            K.clear_session()
            gc.collect()
            eval_end = time.time()

            print('{} out of {}: {} run time  {}'.format(instance + 1, total_model_count, model_path,
                                                         eval_end - eval_start))

        final_ensemble_predicted_class = np.argmax(ensemble_labels_hot_store, axis=1)
        ensemble_labels_hot_store = np.array(ensemble_labels_hot_store) / total_model_count
        return ensemble_predicted_class_store, final_ensemble_predicted_class, ensemble_labels_hot_store
