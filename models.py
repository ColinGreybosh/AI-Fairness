# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

import numpy as np


class BaseModel(object):

    def __init__(self):
        pass

    def bare_model(self):
        pass

    def train(self):
        pass



class LRModel(BaseModel):

    model_name = 'Logistic Regression'

    def train (self, dataset_train, SCALER):
        print ('[INFO]: training logistic regression')
        if SCALER:
            model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
        else:
            model = make_pipeline(LogisticRegression(solver='liblinear', random_state=1))
        fit_params = {'logisticregression__sample_weight': dataset_train.instance_weights}
        model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)

        return model

    def bare_model(self):

        model = LogisticRegression(solver='liblinear', random_state=1)

        return model


class RFModel(BaseModel):

    model_name = 'Random Forest'
    def __init__(self, n_est = 1000, min_leaf=5):
        self.n_est = n_est
        self.min_leaf = min_leaf

    def train (self, dataset_train, SCALER):
        print ('[INFO]: training random forest')
        if SCALER:
            model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=self.n_est, min_samples_leaf=self.min_leaf))
        else:
            model = make_pipeline(RandomForestClassifier(n_estimators=self.n_est, min_samples_leaf=self.min_leaf))
        fit_params = {'randomforestclassifier__sample_weight': dataset_train.instance_weights}
        model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)

        return model

    def bare_model(self):

        model = RandomForestClassifier(n_estimators=self.n_est, min_samples_leaf=self.min_leaf)

        return model


class SVMModel(BaseModel):

    model_name = 'SVM'

    def train (self, dataset_train, SCALER):
        print ('[INFO]: training svm')
        if SCALER:
            model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        else:
            model = make_pipeline(SVC(gamma='auto', probability=True))
        fit_params = {'svc__sample_weight': dataset_train.instance_weights}
        model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)

        return model

    def bare_model(self):

        model = SVC(gamma='auto',probability=True)

        return model


class NNModel(BaseModel):

    model_name = 'Neural Network'

    def train (self, dataset_train, SCALER):
        print ('[INFO]: training neural network')
        if SCALER:
            model = make_pipeline(scaler, MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        else:
            model = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        fit_params = {'mlpclassifier__sample_weight': dataset_train.instance_weights}
        model.fit(dataset_train.features, dataset_train.labels.ravel())

        return model

    def bare_model(self):

        model = MLPClassifierWrapper(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        return model


class NBModel(BaseModel):

    model_name = 'Gaussian NB'

    def train (self, dataset_train, SCALER):
        print ('[INFO]: training Gaussian nb')
        if SCALER:
            model = make_pipeline(scaler, GaussianNB())
        else:
            model = make_pipeline(GaussianNB())
        fit_params = {'gaussiannb__sample_weight': dataset_train.instance_weights}
        model.fit(dataset_train.features, dataset_train.labels.ravel(), **fit_params)

        return model

    def bare_model(self):

        model = GaussianNB()

        return model


class TModel():

    def __init__(self, model_type):
        self.model_type = model_type

    def set_model(self, dataset, SCALER):
        if self.model_type == 'lr':
            model = LRModel() 
        elif self.model_type == 'rf':
            model = RFModel() 
        elif self.model_type == 'svm':
            model = SVMModel() 
        elif self.model_type == 'nn':
            model = NNModel() 
        elif self.model_type == 'nb':
            model = NBModel() 
        trained_model = model.train(dataset, SCALER)

        return trained_model


    def get_model(self):
        if self.model_type == 'lr':
            model = LRModel() 
        elif self.model_type == 'rf':
            model = RFModel() 
        elif self.model_type == 'svm':
            model = SVMModel() 
        elif self.model_type == 'nn':
            model = NNModel() 
        elif self.model_type == 'nb':
            model = NBModel() 
        returned_model = model.bare_model()

        return returned_model


class MLPClassifierWrapper(MLPClassifier):

    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        #sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)
        sample_weight = sample_weight / np.sum(sample_weight.values,dtype=np.float64)
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()

        #X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        X_train_resampled = np.zeros((X_train.shape), dtype=np.float32)
        #y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        y_train_resampled = np.zeros((y_train.shape), dtype=np.int)
        for i in range(X_train.shape[0]):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(X_train.shape[0]), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


