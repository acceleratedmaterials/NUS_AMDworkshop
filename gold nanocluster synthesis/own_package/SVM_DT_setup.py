from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, merge, Input
from keras import regularizers
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, f1_score, matthews_corrcoef, mean_squared_error
from sklearn import tree
from sklearn.externals import joblib
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import time
# NGF
from .NGF.models import build_graph_conv_net_fp_only

class SVM:
    def __init__(self, hparams, fl):
        """
        Initialises new linear SVM model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_dim
        self.labels_dim = fl.n_classes
        self.hparams = hparams

        # Left side
        lc = Input(shape=(fl.features_c_dim,))

        # Input concat vector into SVM
        prediction = Dense(2, activation='linear')(lc)

        self.model = Model(
            input=lc,
            output=prediction)

        # SVM uses categorical hinge loss
        self.model.compile(optimizer='Adam', loss="categorical_hinge", metrics=['accuracy'])

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


class SVM_smiles:
    def __init__(self, hparams, fl):
        """
        Initialises new linear SVM model based on input features_dim, labels_dim, hparams
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

        # Input concat vector into SVM
        prediction = Dense(2, activation='linear')(left_combined)

        self.model = Model(
            input=[lc] + [left_tensor for molecule in left_features_d for left_tensor in molecule],
            output=prediction)

        # SVM uses categorical hinge loss
        self.model.compile(optimizer='Adam', loss="categorical_hinge", metrics=['accuracy'])

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


def create_dt_hparams(max_depth=None, min_samples_split=2):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    names = ['max_depth','min_samples_split']
    values = [max_depth, min_samples_split]
    hparams = dict(zip(names, values))
    return hparams


class DT_classifer:
    def __init__(self,hparams):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes: hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        # Build New Compiled DNN model
        self.hparams=hparams
        self.model = tree.DecisionTreeClassifier(splitter ='random',max_depth = hparams['max_depth'] ,min_samples_split=hparams['min_samples_split'])

    def train_model(self, fl,
                    save_name='DT', save_dir='./save/models/',
                    plot_mode=False, save_mode=False):
        # Training model
        training_features = fl.features_c_a  # for decision tree, no need to norm features_c
        training_labels = fl.labels
        self.model.fit(training_features, training_labels)

        # Saving Model
        if save_mode:
            joblib.dump(self.model, filename= save_dir + save_name + 'pkl')
        # Plotting
        if plot_mode:
            dot_data=tree.export_graphviz(self.model,out_file=None,
                                          feature_names=fl.features_c_names,
                                          filled=True, rounded=True)
            graph=graphviz.Source(dot_data)
            graph.render(filename='./plots/'+save_name)
        return self.model

    def eval(self, eval_fl):
        features = eval_fl.features_c_a
        labels = eval_fl.labels
        predictions_class = self.model.predict(features)
        test = self.model.predict_proba(features)

        # Calculating metrics
        acc = accuracy_score(labels, predictions_class)
        cm = confusion_matrix(labels, predictions_class)
        try:
            f1s = f1_score(labels, predictions_class)  # Will work for binary classification
        except ValueError:  # Multi-class will raise ValueError from sklearn f1_score function
            f1s = f1_score(labels, predictions_class, average='micro')  # Must use micro averaging for multi-class
        mcc = matthews_corrcoef(labels, predictions_class)
        return predictions_class, acc, cm, f1s, mcc,test
