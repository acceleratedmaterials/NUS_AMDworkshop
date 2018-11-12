from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, merge, Input
from keras import regularizers
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, f1_score, matthews_corrcoef, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import time
# NGF
from .NGF.models import build_graph_conv_net_fp_only


class DNN_classifer:
    def __init__(self, hparams, fl):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim = fl.n_classes
        self.hparams = hparams

        # Build New Compiled DNN model
        self.model = self.create_DNN_model()
        self.model.compile(optimizer=hparams['optimizer'], loss='categorical_crossentropy')

    def create_DNN_model(self):
        """
        Creates Keras Dense Neural Network model. Not compiled yet!
        :return: Uncompiled Keras DNN model
        """
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        numel = len(hidden_layers)
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='sigmoid'))
        return model

    def train_model(self, fl,
                    save_name='cDNN_training_only.h5', save_dir='./save/models/',
                    plot_mode=False, save_mode=False):
        # Training model
        training_features = fl.features_c_norm_a
        training_labels = fl.labels_hot
        history = self.model.fit(training_features, training_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        return self.model

    def eval(self, eval_fl):
        eval_start = time.time()
        features = eval_fl.features_c_norm_a
        labels = eval_fl.labels
        labels_hot = eval_fl.labels_hot
        predictions = self.model.predict(features)
        predictions_class = [predicted_labels_hot.index(max(predicted_labels_hot)) for predicted_labels_hot in
                             np.ndarray.tolist(predictions)]
        # Calculating metrics
        acc = accuracy_score(labels, predictions_class)
        ce = log_loss(labels_hot, predictions)
        cm = confusion_matrix(labels, predictions_class)
        try:
            f1s = f1_score(labels, predictions_class)  # Will work for binary classification
        except ValueError:  # Multi-class will raise ValueError from sklearn f1_score function
            f1s = f1_score(labels, predictions_class, average='micro')  # Must use micro averaging for multi-class
        mcc = matthews_corrcoef(labels, predictions_class)

        eval_end = time.time()
        print('eval run time : {}'.format(eval_end - eval_start))
        return predictions_class, acc, ce, cm, f1s, mcc


class DNN:
    def __init__(self, hparams, fl):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim = 1
        self.hparams = hparams

        # Build New Compiled DNN model
        self.model = self.create_DNN_model()
        self.model.compile(optimizer=hparams['optimizer'], loss='mse')

    def create_DNN_model(self):
        """
        Creates Keras Dense Neural Network model. Not compiled yet!
        :return: Uncompiled Keras DNN model
        """
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        numel = len(hidden_layers)
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='linear'))
        return model

    def train_model(self, fl,
                    save_name='cDNN_training_only.h5', save_dir='./save/models/',
                    plot_mode=False, save_mode=False):
        # Training model
        training_features = fl.features_c_norm_a
        training_labels = fl.labels
        history = self.model.fit(training_features, training_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        return self.model

    def eval(self, eval_fl):
        eval_start = time.time()
        features = eval_fl.features_c_norm_a

        labels = eval_fl.labels
        labels_hot = eval_fl.labels_hot
        predictions = self.model.predict(features)
        predictions_class = [predicted_labels_hot.index(max(predicted_labels_hot)) for predicted_labels_hot in
                             np.ndarray.tolist(predictions)]
        # Calculating metrics
        acc = accuracy_score(labels, predictions_class)
        ce = log_loss(labels_hot, predictions)
        cm = confusion_matrix(labels, predictions_class)
        try:
            f1s = f1_score(labels, predictions_class)  # Will work for binary classification
        except ValueError:  # Multi-class will raise ValueError from sklearn f1_score function
            f1s = f1_score(labels, predictions_class, average='micro')  # Must use micro averaging for multi-class
        mcc = matthews_corrcoef(labels, predictions_class)

        eval_end = time.time()
        print('eval run time : {}'.format(eval_end - eval_start))
        return predictions_class, acc, ce, cm, f1s, mcc


class DNN_classifer_smiles:
    def __init__(self, hparams, fl):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.features_d_count = fl.features_d_count
        self.labels_dim = fl.n_classes
        self.hparams = hparams
        self.fp_length = [hparams['fp_length'] for _ in range(hparams['fp_number'])]
        self.conv_width = [hparams['conv_width'] for _ in range(hparams['conv_number'])]

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
                                                              conv_activation='relu', fp_activation='softmax')
                                 )
        # Concat left side 4 inputs. Unlike SNN, there is no right side.
        left_combined = merge.Concatenate()([lc] + left_conv_net)

        # Input concat vector into single DNN net with single node classification output (1 node output only).
        prediction = self.single_DNN_net()(left_combined)

        self.model = Model(
            input=[lc] + [left_tensor for molecule in left_features_d for left_tensor in molecule],
            output=prediction)

        self.model.compile(optimizer=hparams['optimizer'], loss='categorical_crossentropy')

    def single_DNN_net(self):
        """
        Creates Keras Dense Neural Network model. Not compiled yet!
        :return: Uncompiled Keras DNN model
        """
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim + self.features_d_count * self.hparams['fp_length'],
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        numel = len(hidden_layers)
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='sigmoid'))
        return model

    def train_model(self, fl,
                    save_name='cDNN_training_only.h5', save_dir='./save/models/',
                    plot_mode=False, save_mode=False):
        # Training model
        features_c_norm_a = fl.features_c_norm_a
        features_d_a = fl.features_d_a
        features=[[] for _ in range(1 + self.features_d_count * 3)]
        # Append features c to the 0th list in the pairs nested list
        idx = 0
        features[idx]=features_c_norm_a
        # Add one to the counter idx that is keeping track of the current position in the nested list
        idx += 1
        # Adding information about features_d
        for single_molecule in features_d_a:
            for single_tensor in single_molecule:
                features[idx]=single_tensor
                idx += 1
        labels = fl.labels_hot
        history = self.model.fit(features, labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        return self.model

    def eval(self, eval_fl):
        eval_start = time.time()
        features_c_norm_a = eval_fl.features_c_norm_a
        features_d_a = eval_fl.features_d_a
        features=[[] for _ in range(1 + self.features_d_count * 3)]
        # Append features c to the 0th list in the pairs nested list
        idx = 0
        features[idx]=features_c_norm_a
        # Add one to the counter idx that is keeping track of the current position in the nested list
        idx += 1
        # Adding information about features_d
        for single_molecule in features_d_a:
            for single_tensor in single_molecule:
                features[idx]=single_tensor
                idx += 1
        labels = eval_fl.labels
        labels_hot = eval_fl.labels_hot
        predictions = self.model.predict(features)
        predictions_class = [predicted_labels_hot.index(max(predicted_labels_hot)) for predicted_labels_hot in
                             np.ndarray.tolist(predictions)]
        # Calculating metrics
        acc = accuracy_score(labels, predictions_class)
        ce = log_loss(labels_hot, predictions)
        cm = confusion_matrix(labels, predictions_class)
        try:
            f1s = f1_score(labels, predictions_class)  # Will work for binary classification
        except ValueError:  # Multi-class will raise ValueError from sklearn f1_score function
            f1s = f1_score(labels, predictions_class, average='micro')  # Must use micro averaging for multi-class
        mcc = matthews_corrcoef(labels, predictions_class)

        eval_end = time.time()
        print('eval run time : {}'.format(eval_end - eval_start))
        return predictions_class, acc, ce, cm, f1s, mcc


class DNN_smiles:
    def __init__(self, hparams, fl):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim = 1  # For classification for SMILES
        self.hparams = hparams
        self.fp_length = [hparams['fp_length'] for _ in range(hparams['fp_number'])]
        self.conv_width = [hparams['conv_width'] for _ in range(hparams['conv_number'])]

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
                                                              conv_activation='relu', fp_activation='softmax')
                                 )
        # Concat left side 4 inputs. Unlike SNN, there is no right side.
        left_combined = merge.Concatenate()([lc] + left_conv_net)

        # Input concat vector into single DNN net with single node classification output (1 node output only).
        prediction = self.single_DNN_net()(left_combined)

        self.model = Model(
            input=[lc] + [left_tensor for molecule in left_features_d for left_tensor in molecule],
            output=prediction)

        self.model.compile(optimizer=hparams['optimizer'], loss='mse')

    def single_DNN_net(self):
        """
        Creates Keras Dense Neural Network model. Not compiled yet!
        :return: Uncompiled Keras DNN model
        """
        model = Sequential()
        hidden_layers = self.hparams['hidden_layers']
        generator_dropout = self.hparams.get('dropout', 0)
        model.add(Dense(hidden_layers[0],
                        input_dim=self.features_dim,
                        activation=self.hparams['activation'],
                        kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        numel = len(hidden_layers)
        if generator_dropout != 0:
            model.add(Dropout(generator_dropout))
        if numel > 1:
            if hidden_layers[1] != 0:  # Even if hidden layers has 2 elements, 2nd element may be 0
                for i in range(numel - 1):
                    model.add(Dense(hidden_layers[i + 1],
                                    activation=self.hparams['activation'],
                                    kernel_regularizer=regularizers.l2(self.hparams['reg_term'])))
        model.add(Dense(self.labels_dim, activation='linear'))
        return model

    def train_model(self, fl,
                    save_name='cDNN_training_only.h5', save_dir='./save/models/',
                    plot_mode=False, save_mode=False):
        # Training model
        training_features = fl.features_c_norm_a
        training_labels = fl.labels
        history = self.model.fit(training_features, training_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_mode:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        return self.model

    def eval(self, eval_fl):
        eval_start = time.time()
        features = eval_fl.features_c_norm_a
        labels = eval_fl.labels
        predictions = self.model.predict(features)
        # Calculating metrics
        mse = mean_squared_error(labels, predictions)
        eval_end = time.time()
        print('eval run time : {}'.format(eval_end - eval_start))
        return predictions, mse
