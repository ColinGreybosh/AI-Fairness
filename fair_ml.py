import sys
import random
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN
import numpy as np
from collections import defaultdict
import pandas as pd
import statistics

# Datasets
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21
from aif360.datasets import BinaryLabelDataset

# Metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics import utils 

# optimizer
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

# Scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

#Bias mitigation techniques
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, OptimPreproc, Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing, ARTClassifier, GerryFairClassifier, MetaFairClassifier, PrejudiceRemover 
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing\
        import CalibratedEqOddsPostprocessing 
from aif360.algorithms.postprocessing.eq_odds_postprocessing\
        import EqOddsPostprocessing
#from aif360.algorithms.postprocessing.reject_option_classification\
#        import RejectOptionClassification
from aif360.algorithms.postprocessing import RejectOptionClassification
from common_utils import compute_metrics

#from myutils import balance
from oversample import synthetic
from oversample import label_bias, selection_bias 
from plot_baseline import plot_baseline

# construct argument parser
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, help="dataset: adult, compas, german, bank, meps19, grade ")
ap.add_argument("-c", "--classifier", required=True, help="baseline model: lr, rf, svm, nn")
ap.add_argument("-m", "--mitigator", required=False, help="mitigators: dir, rew, egr, pr, cpp, ro, all")
ap.add_argument("-b", "--bias", required=True, help="amount of bias: o-1")
ap.add_argument("-o", "--os", required=False, help="oversample mode: 1: privi unfav 2: unpriv fav")
args = vars(ap.parse_args())

DATASET = args["data"]
BASELINE = args["classifier"]
if not args["mitigator"]:
    # mitigator to compare with
    MITIGATOR = 'all'
else:
    MITIGATOR = args["mitigator"]

BIAS = float(args["bias"])
if not args["os"]:
    #defautl oversampleing mode: unprivileged favor
    OS_MODE = 2 
else:
    OS_MODE = int(args["os"])

# global constants
if BASELINE == 'svm' or BASELINE == 'nn':
    SCALER = True 
else:
    SCALER = False 
DISPLAY = False 
# grid search threshold
THRESH_ARR = 0.5

# random forest parameters
n_est = 1000
min_leaf = 5

# load data 'adult', 'grade', 'bank', 'german', 'compas', or 'meps'
if DATASET == 'adult':
    protected_attribute_used = 1
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_adult(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_adult(['race'])

    optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }
    sens_attr = dataset_orig.protected_attribute_names[0]
elif DATASET == 'grade':
    #load dataset and print shape
    dataset_loc = "./student/student-por.csv"
    df = pd.read_csv(dataset_loc, sep=";")
    print('Dataset consists of {} Observations and {} Variables'.format(df.shape[0],df.shape[1]))
    df.drop(['G1', 'G2'], inplace=True, axis=1)
    features = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
           'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
           'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
           'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
           'Walc', 'health']
    labels = ['absences', 'G3']
    df['sex'] = df['sex'].map( {'F':0, 'M':1})
    df['Pstatus'] = df['Pstatus'].map( {'A':0, 'T':1})
    df['age'].values[df['age'] < 18] = 0
    df['age'].values[df['age'] >= 18] = 1
    df['G3'].values[df['G3'] < 12] = 0
    df['G3'].values[df['G3'] >= 12] = 1
    df['G3'].unique()
    df['absences'].values[df['absences'] <= 4] = 1
    df['absences'].values[df['absences'] > 4] = 0
    df['absences'].unique()

    numvar = [key for key in dict(df.dtypes)
                       if dict(df.dtypes)[key]
                           in ['float64','float32','int32','int64']] # Numeric Variable

    catvar = [key for key in dict(df.dtypes)
                 if dict(df.dtypes)[key] in ['object'] ] # Categorical Varible
    for cat in catvar:
        df[cat] = LabelEncoder().fit_transform(df[cat])

    proclsvars = ['sex', 'Pstatus', 'age']
    depenvars = ['G3', 'absences']

    proclsvar = 'sex'
    depenvar = 'G3'

    privileged_groups = [{proclsvar: 1}]
    unprivileged_groups = [{proclsvar: 0}]
    favorable_label = 0 
    unfavorable_label = 1
    dataset_orig = BinaryLabelDataset(favorable_label=favorable_label,
                            unfavorable_label=unfavorable_label,
                            df=df,
                            label_names=[depenvar],
                            protected_attribute_names=[proclsvar],
                            unprivileged_protected_attributes=unprivileged_groups)
    sens_attr = dataset_orig.protected_attribute_names[0]

elif DATASET == 'bank':
    dataset_orig = BankDataset(
        protected_attribute_names=['age'],           # this dataset also contains protected
        privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    )
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    sens_attr = dataset_orig.protected_attribute_names[0]

elif DATASET == 'german':
    # 1:age ,2: foreign
    protected_attribute_used = 1
    if protected_attribute_used == 1:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = GermanDataset(
            protected_attribute_names=['age'],           # this dataset also contains protected
                                                         # attribute for "sex" which we do not
                                                         # consider in this evaluation
            privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
            features_to_drop=['personal_status','sex'], # ignore sex-related attributes
        )
        #dataset_orig = load_preproc_data_german(['age'])
        optim_options = {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.1,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }    

    else:
        privileged_groups = [{'foreign': 1}]
        unprivileged_groups = [{'foreign': 0}]

        default_mappings = {
            'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
            'protected_attribute_maps': [{1.0: 'No', 0.0: 'Yes'}]
        }

        categorical_features=['status', 'credit_history', 'purpose',
                             'savings', 'employment', 'other_debtors', 'property',
                             'installment_plans', 'housing', 'skill_level', 'telephone']

        def default_preprocessing(df):
            """Adds a derived sex attribute based on personal_status."""

            status_map = {'A201': 'Yes', 'A202': 'No'} 
            df['foreign'] = df['foreign_worker'].replace(status_map)

            return df

        dataset_orig = GermanDataset(
            protected_attribute_names=['foreign'],       # this dataset also contains protected
                                                         # attribute for "sex" which we do not
                                                         # consider in this evaluation
            privileged_classes=[['No']],                 # none foreign is considered privileged
            features_to_drop=['personal_status', 'foreign_worker'], # ignore sex-related attributes
            categorical_features=categorical_features,
            custom_preprocessing=default_preprocessing,
            metadata=default_mappings
        )
        optim_options = {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.1,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }

    sens_attr = dataset_orig.protected_attribute_names[0]


elif DATASET == 'compas':
    dataset_orig = load_preproc_data_compas()

    #dataset_orig = CompasDataset(
    #    protected_attribute_names=['race'],           # this dataset also contains protected
                                                     # attribute for "sex" which we do not
                                                     # consider in this evaluation
    #    privileged_classes=[['Caucasian']],      # race Caucasian is considered privileged
    #    features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
    #)

    sens_attr = dataset_orig.protected_attribute_names[0]
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

else: 
    if DATASET == 'meps19':
        dataset_orig = MEPSDataset19()
    elif DATASET == 'meps20':
        dataset_orig = MEPSDataset20()
    else: 
        dataset_orig = MEPSDataset21()

    sens_ind = 0
    sens_attr = dataset_orig.protected_attribute_names[sens_ind]

    sens_attr = dataset_orig.protected_attribute_names[0]
    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_orig.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_orig.privileged_protected_attributes[sens_ind]]



def test(dataset, model, thresh_arr, metric_arrs):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        neg_ind = np.where(model.classes_ == dataset.unfavorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
        neg_ind = 1
        #print('y_val_pre_prob: ', y_val_pred_prob)

    if metric_arrs is None:
        metric_arrs = defaultdict(list)

    for thresh in thresh_arr:
        #y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)  
        y_val_pred = np.array([0]*y_val_pred_prob.shape[0])
        y_val_pred[np.where(y_val_pred_prob[:,pos_ind] > thresh)[0]] = f_label
        y_val_pred[np.where(y_val_pred_prob[:,pos_ind] <= thresh)[0]] = uf_label
        y_val_pred = y_val_pred.reshape(-1,1)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(1 - min((metric.disparate_impact()), 1/metric.disparate_impact()))
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
        #metric_arrs['false_positive_rate'].append(metric.false_positive_rate(privileged=False))
        #metric_arrs['false_negative_rate'].append(metric.false_negative_rate(privileged=False))

    return metric_arrs


def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)

def describe_metrics(metrics, thresh_arr, TEST=True):
    if not TEST: 
        best_ind = np.argmax(metrics['bal_acc'])
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    else:
        best_ind = -1
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
#    print("Corresponding false positive_rate: {:6.4f}".format(metrics['false_positive_rate'][best_ind]))
#    print("Corresponding false negative_rate: {:6.4f}".format(metrics['false_negative_rate'][best_ind]))


def get_test_metrics(dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, test_metrics, SCALER):

    dataset = dataset_orig_train
    scaler = MinMaxScaler(feature_range= (0,1))

    # fitting the model
    if model_type == 'lr':
        if SCALER:
            model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
        else:
            model = make_pipeline(LogisticRegression(solver='liblinear', random_state=1))
        fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

    elif model_type == 'rf':
        if SCALER:
            model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf))
        else:
            model = make_pipeline(RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf))
        fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
    elif model_type == 'svm':
        if DATASET == 'adult' or DATASET == 'bank':
            Kernel = 'linear'
        else:
            Kernel = 'rbf'

        if SCALER:
            model = make_pipeline(StandardScaler(), SVC(kernel=Kernel,gamma='auto', probability=True))
        else:
            model = make_pipeline(SVC(kernel=Kernel, gamma='auto', probability=True))
        fit_params = {'svc__sample_weight': dataset.instance_weights}
    elif model_type == 'nn':
        if SCALER:
            model = make_pipeline(scaler, MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        else:
            model = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        fit_params = {'mlpclassifier__sample_weight': dataset.instance_weights}


    if model_type == 'nn':
        mod_orig = model.fit(dataset.features, dataset.labels.ravel())
    else:
        mod_orig = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

    thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
    
    # find the best threshold for balanced accuracy
    print('Validating Original ...')
    if SCALER:
        scale_orig = StandardScaler()
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_val_pred.features = scale_orig.fit_transform(dataset_orig_val_pred.features)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.features = scale_orig.fit_transform(dataset_orig_test_pred.features)
    else:
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    val_metrics = test(dataset=dataset_orig_val_pred,
                       model=mod_orig,
                       thresh_arr=thresh_arr, metric_arrs=None)
    orig_best_ind = np.argmax(val_metrics['bal_acc'])

    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

    if DISPLAY:
        plot(thresh_arr, model_type + ' Original Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             disp_imp_err, '1 - min(DI, 1/DI)')

        plot(thresh_arr, model_type + ' Original Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             val_metrics['avg_odds_diff'], 'avg. odds diff.')

        plt.show()


    print('Testing Original ...')
    test_metrics = test(dataset=dataset_orig_test_pred,
                               model=mod_orig,
                               thresh_arr=[thresh_arr[orig_best_ind]], metric_arrs=test_metrics)

    describe_metrics(test_metrics, thresh_arr)

    return test_metrics


def orig_no_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, SCALER):

    dataset = dataset_orig_train
    orig_metrics = get_test_metrics(dataset, dataset_orig_val, dataset_orig_test, model_type, orig_metrics, SCALER)

    return orig_metrics


def synth_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, model_type, transf_metrics, f_label, uf_label, SCALER):

    # generating synthetic data 
    dataset_transf_train = synthetic(dataset_orig_train, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, f_label, uf_label, os_mode = OS_MODE)
    print('origin, transf: ', dataset_orig_train.features.shape[0], dataset_transf_train.features.shape[0])

    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)
    print('after transf priv: ', metric_transf_train.base_rate(privileged=True))
    print('after transf unpriv: ', metric_transf_train.base_rate(privileged=False))
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())


    # fitting the model on the transformed dataset with synthetic generator
    dataset = dataset_transf_train
    transf_metrics = get_test_metrics(dataset, dataset_orig_val, dataset_orig_test, model_type, transf_metrics, SCALER)

    return metric_transf_train, transf_metrics 


# disparity impact remover (dir) mitigator (pre-processing)
def dir_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test,  sensitive_attribute, model_type, dir_metrics, SCALER):

    DIR = DisparateImpactRemover(sensitive_attribute=sensitive_attribute)
    dataset_dir_train = DIR.fit_transform(dataset_orig_train)
    dataset_dir_val = DIR.fit_transform(dataset_orig_val)
    dataset_dir_test = DIR.fit_transform(dataset_orig_test)

    dataset = dataset_dir_train

    dir_metrics = get_test_metrics(dataset, dataset_dir_val, dataset_dir_test, model_type, dir_metrics, SCALER)

    return dir_metrics


# Optim Prepproc (op) mitigator
def op_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test,  sensitive_attribute, model_type, op_metrics, SCALER):

    OP = OptimPreproc(unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups)
    dataset_op_train = OP.fit_transform(dataset_orig_train)
    return op_metrics


def reweigh_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test,  unprivileged_groups, privileged_groups, model_type, reweigh_metrics, SCALER):

    # transform the data with preprocessing reweighing and fit the model
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_reweigh_train = RW.fit_transform(dataset_orig_train)

    dataset = dataset_reweigh_train
    if model_type == 'lr':
        if SCALER:
            model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
        else:
            model = make_pipeline(LogisticRegression(solver='liblinear', random_state=1))
        fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
    elif model_type == 'rf':
        if SCALER:
            model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf))
        else:
            model = make_pipeline(RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf))
        fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
    elif model_type == 'svm':
        if DATASET == 'adult' or DATASET == 'bank':
            Kernel = 'linear'
        else:
            Kernel = 'rbf'
        if SCALER:
            model = make_pipeline(StandardScaler(), SVC(kernel=Kernel, gamma='auto',probability=True))
        else:
            model = make_pipeline(SVC(kernel=Kernel, gamma='auto',probability=True))
        fit_params = {'svc__sample_weight': dataset.instance_weights}
    elif model_type == 'nn':
        if SCALER:
            model = make_pipeline(StandardScaler(), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        else:
            model = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
        fit_params = {'mlpclassifier__sample_weight': dataset.instance_weights}


    if model_type == 'nn':
        mod_reweigh = model.fit(dataset.features, dataset.labels.ravel())
    else:
        mod_reweigh = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

    thresh_arr = np.linspace(0.01, THRESH_ARR, 50)
    if SCALER:
        scale_orig = StandardScaler()
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_val_pred.features = scale_orig.fit_transform(dataset_orig_val_pred.features)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.features = scale_orig.fit_transform(dataset_orig_test_pred.features)
    else:
        dataset_orig_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    val_metrics = test(dataset=dataset_orig_val_pred,
                       model=mod_reweigh,
                       thresh_arr=thresh_arr, metric_arrs=None)
    reweigh_best_ind = np.argmax(val_metrics['bal_acc'])

    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

    if DISPLAY:
        plot(thresh_arr, model_type + ' Reweighing Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             disp_imp_err, '1 - min(DI, 1/DI)')

        plot(thresh_arr, model_type + ' Reweighing Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             val_metrics['avg_odds_diff'], 'avg. odds diff.')
        plt.show()

    #describe_metrics(val_metrics, thresh_arr)

    reweigh_metrics = test(dataset=dataset_orig_test_pred,
                           model=mod_reweigh,
                           thresh_arr=[thresh_arr[reweigh_best_ind]], metric_arrs=reweigh_metrics)

    describe_metrics(reweigh_metrics, thresh_arr)
    
    return reweigh_metrics

class MLPClassifierWrapper(MLPClassifier):

    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        #sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)
        sample_weight = sample_weight / np.sum(sample_weight.values,dtype=np.float64)
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()

        X_train_resampled = np.zeros((X_train.shape), dtype=np.float32)
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

# Exponentiated Gradiant Reduction (In-processing)
def egr_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, egr_metrics, model_type, SCALER):

    # set up dataset
    if SCALER:
        egr_orig_scaler = StandardScaler()
        dataset = dataset_orig_train.copy(deepcopy=True)
        dataset.features = egr_orig_scaler.fit_transform(dataset.features)

        dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_val_pred.features = egr_orig_scaler.transform(dataset_val_pred.features)

        dataset_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_test_pred.features = egr_orig_scaler.transform(dataset_test_pred.features)

    else:
        dataset = dataset_orig_train.copy(deepcopy=True)
        dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_test_pred = dataset_orig_test.copy(deepcopy=True)

    dataset.labels = dataset.labels.ravel()
    # set up baseline machine learning model 
    if model_type == 'lr':
        model = LogisticRegression(solver='liblinear', random_state=1)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf)
    elif model_type == 'svm':
        if DATASET == 'adult' or DATASET == 'bank':
            model = SVC(kernel='linear', gamma='auto',probability=True)
        else:
            model = SVC(gamma='auto',probability=True)
    elif model_type == 'nn':
        model = MLPClassifierWrapper(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    elif model_type == 'nb':
        model = GaussianNB() 

    print('fitting EGR ...')
    egr_mod = ExponentiatedGradientReduction(estimator=model, 
                                             T=5,
                                             eps=0.01,
                                             constraints="EqualizedOdds",
                                             drop_prot_attr=False)
    egr_mod.fit(dataset)
    print('Done fitting EGR ...')

    #validate
    thresh_arr = np.linspace(0.01, THRESH_ARR, 50)

    val_metrics = test(dataset=dataset_val_pred,
                       model=egr_mod,
                       thresh_arr=thresh_arr, metric_arrs=None)
    egr_best_ind = np.argmax(val_metrics['bal_acc'])

    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

    if DISPLAY:
        plot(thresh_arr, 'Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             disp_imp_err, '1 - min(DI, 1/DI)')

        plot(thresh_arr, 'Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             val_metrics['avg_odds_diff'], 'avg. odds diff.')

        plt.show()

    egr_metrics = test(dataset=dataset_test_pred,
                       model=egr_mod,
                       thresh_arr=[thresh_arr[egr_best_ind]], metric_arrs=egr_metrics)

    describe_metrics(egr_metrics, [thresh_arr[egr_best_ind]])

    #exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)
    return egr_metrics



def pr_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, pr_orig_metrics, SCALER):

    SCALER = True

    # train 
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=2.0)
    if SCALER:
        pr_orig_scaler = StandardScaler()
        dataset = dataset_orig_train.copy(deepcopy=True)
        dataset.features = pr_orig_scaler.fit_transform(dataset.features)

        dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_val_pred.features = pr_orig_scaler.transform(dataset_val_pred.features)

        dataset_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_test_pred.features = pr_orig_scaler.transform(dataset_test_pred.features)

    else:
        dataset = dataset_orig_train.copy(deepcopy=True)
        dataset_val_pred = dataset_orig_val.copy(deepcopy=True)
        dataset_test_pred = dataset_orig_test.copy(deepcopy=True)

    pr_orig = model.fit(dataset)

    #validate 
    thresh_arr = np.linspace(0.01, THRESH_ARR, 50)

    val_metrics = test(dataset=dataset_val_pred,
                       model=pr_orig,
                       thresh_arr=thresh_arr, metric_arrs=None)
    pr_orig_best_ind = np.argmax(val_metrics['bal_acc'])

    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

    if DISPLAY:
        plot(thresh_arr, 'Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             disp_imp_err, '1 - min(DI, 1/DI)')

        plot(thresh_arr, 'Classification Thresholds',
             val_metrics['bal_acc'], 'Balanced Accuracy',
             val_metrics['avg_odds_diff'], 'avg. odds diff.')

        plt.show()

    pr_orig_metrics = test(dataset=dataset_test_pred,
                           model=pr_orig,
                           thresh_arr=[thresh_arr[pr_orig_best_ind]], metric_arrs=pr_orig_metrics)

    describe_metrics(pr_orig_metrics, [thresh_arr[pr_orig_best_ind]])

    return pr_orig_metrics


def cpp_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, cpp_orig_metrics, model_type, SCALER):

    cost_constraint = "weighted" # "fnr", "fpr", "weighted"

    # Verify metric name
    allowed_constraints = ["fnr", "fpr", "weighted"]
    if cost_constraint not in allowed_constraints:
        raise ValueError("Constraint should be one of allowed constraints")

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)

    if SCALER:
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)

        dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
        X_valid = scale_orig.transform(dataset_orig_valid_pred.features)

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = scale_orig.transform(dataset_orig_test_pred.features)

    else:
        X_train = dataset_orig_train.features

        dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
        X_valid = dataset_orig_valid_pred.features

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = dataset_orig_test_pred.features

    y_train = dataset_orig_train.labels.ravel()

    # Logistic regression classifier and predictions
    if model_type == 'lr':
        lmod = LogisticRegression(solver='liblinear', random_state=1)
    elif model_type == 'rf':
        lmod = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf)
    elif model_type == 'svm':
        if DATASET == 'adult' or DATASET == 'bank':
            lmod = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto',probability=True))
        else:
            lmod = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
    elif model_type == 'nn':
        lmod = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))

    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred.labels = y_train_pred
    y_valid = dataset_orig_valid_pred.labels.ravel()
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    y_test = dataset_orig_test_pred.labels.ravel()
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 50
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, THRESH_ARR, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
    
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
    
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_val,
                                                 dataset_orig_valid_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                           +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    # Learn parameters to equalize odds and apply to create a new dataset
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                         unprivileged_groups = unprivileged_groups,
                                         cost_constraint=cost_constraint,
                                         seed=None)
    cpp = cpp.fit(dataset_orig_val, dataset_orig_valid_pred)


    # Metrics for the validation set
    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    metric_valid_bef = compute_metrics(dataset_orig_val, dataset_orig_valid_pred, 
                       unprivileged_groups, privileged_groups, None, disp=False)

    # Transform the validation set
    dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)

    metric_valid_aft = compute_metrics(dataset_orig_val, dataset_transf_valid_pred, 
                       unprivileged_groups, privileged_groups, None, disp=False)

    # Metrics for the test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                      unprivileged_groups, privileged_groups, None, disp = False)

    # Metrics for the transformed test set
    dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

    cpp_orig_metrics = compute_metrics(dataset_orig_test, dataset_transf_test_pred, 
                      unprivileged_groups, privileged_groups, cpp_orig_metrics, disp=False)

    describe_metrics(cpp_orig_metrics, [best_class_thresh]) #[thresh_arr[pr_orig_best_ind]])

    return cpp_orig_metrics


# Reject Option Mitigator
def ro_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, ro_orig_metrics, model_type, SCALER):

    # Metric used (should be one of allowed_metrics)
    metric_name = "Statistical parity difference"

    # Upper and lower bound on the fairness metric used
    metric_ub = 0.05
    metric_lb = -0.05
        
    # Verify metric name
    allowed_metrics = ["Statistical parity difference",
                       "Average odds difference",
                       "Equal opportunity difference"]
    if metric_name not in allowed_metrics:
        raise ValueError("Metric name should be one of allowed metrics")

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)

    if SCALER:
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)

        dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
        X_valid = scale_orig.transform(dataset_orig_valid_pred.features)

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = scale_orig.transform(dataset_orig_test_pred.features)

    else:
        X_train = dataset_orig_train.features

        dataset_orig_valid_pred = dataset_orig_val.copy(deepcopy=True)
        X_valid = dataset_orig_valid_pred.features

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = dataset_orig_test_pred.features

    y_train = dataset_orig_train.labels.ravel()

    # Logistic regression classifier and predictions
    if model_type == 'lr':
        lmod = LogisticRegression(solver='liblinear', random_state=1)
    elif model_type == 'rf':
        lmod = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=min_leaf)
    elif model_type == 'svm':
        if DATASET == 'adult' or DATASET == 'bank':
            lmod = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto',probability=True))
        else:
            lmod = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
    elif model_type == 'nn':
        lmod = make_pipeline(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))

    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)

    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred.labels = y_train_pred
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 50
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, THRESH_ARR, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
    
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
    
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_val,
                                                 dataset_orig_valid_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                           +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                     privileged_groups=privileged_groups, 
                                     low_class_thresh=0.01, high_class_thresh=THRESH_ARR,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name=metric_name,
                                      metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_val, dataset_orig_valid_pred)

    # Metrics for the validation set
    fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    metric_valid_bef = compute_metrics(dataset_orig_val, dataset_orig_valid_pred, 
                       unprivileged_groups, privileged_groups, None, disp=False)

    # Transform the validation set
    dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)

    metric_valid_aft = compute_metrics(dataset_orig_val, dataset_transf_valid_pred, 
                       unprivileged_groups, privileged_groups, None, disp=False)

    # Metrics for the test set
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred, 
                      unprivileged_groups, privileged_groups, None, disp = False)

    # Metrics for the transformed test set
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

    ro_orig_metrics = compute_metrics(dataset_orig_test, dataset_transf_test_pred, 
                      unprivileged_groups, privileged_groups, ro_orig_metrics, disp=False)

    describe_metrics(ro_orig_metrics, [best_class_thresh]) #[thresh_arr[pr_orig_best_ind]])

    return ro_orig_metrics



###########################
#         Main            # 
###########################

# loop N times 
N = 10 
# percentage of favor and unfavor
priv_metric_orig = defaultdict(float)
favor_metric_orig = defaultdict(float)
favor_metric_transf = defaultdict(float)

orig_metrics = defaultdict(list)
transf_metrics = defaultdict(list) 
reweigh_metrics = defaultdict(list) 
dir_metrics = defaultdict(list) 
egr_metrics = defaultdict(list) 
pr_metrics = defaultdict(list) 
cpp_metrics = defaultdict(list) 
ro_metrics = defaultdict(list) 

p = 0.7
for i in range(N):
    # split dataset into train, validation, and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([p], shuffle=True)
    dataset_orig_val = dataset_orig_test

    # favorable and unfavorable labels and feature_names
    f_label = dataset_orig_train.favorable_label
    uf_label = dataset_orig_train.unfavorable_label
    feature_names = dataset_orig_train.feature_names

    dataset_orig_train = label_bias(dataset_orig_train, unprivileged_groups, BIAS)
    # uncomment the following for selection bias
    #dataset_orig_train = selection_bias(dataset_orig_train, unprivileged_groups, BIAS)

    # show data info
    print("#### Training Dataset shape")
    print(dataset_orig_train.features.shape)
    print("#### Favorable and unfavorable labels")
    print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
    print("#### Protected attribute names")
    print(dataset_orig_train.protected_attribute_names)
    print("#### Privileged and unprivileged protected attribute values")
    print(privileged_groups, unprivileged_groups)
    #print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
    print("#### Dataset feature names")
    print(dataset_orig_train.feature_names)
    print(dataset_orig_train.features[0]) #:,4])

    # check fairness on the original data
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    print("privileged vs. unprivileged: ", metric_orig_train.num_positives(privileged=True) + metric_orig_train.num_negatives(privileged=True), metric_orig_train.num_positives(privileged=False) + metric_orig_train.num_negatives(privileged=False)) 
    base_rate_unprivileged = metric_orig_train.base_rate(privileged=False)
    base_rate_privileged = metric_orig_train.base_rate(privileged=True)
    print('base_pos unpriv: ', base_rate_unprivileged)
    print('base_pos priv: ', base_rate_privileged)
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

    # statistics of favored/positive class BEFORE transf 
    priv_metric_orig['total_priv'] += metric_orig_train.num_instances(privileged = True) 
    priv_metric_orig['total_unpriv'] += metric_orig_train.num_instances(privileged = False) 
    favor_metric_orig['total_favor'] += metric_orig_train.base_rate()
    favor_metric_orig['total_unfavor'] += 1 - metric_orig_train.base_rate()
    favor_metric_orig['priv_favor'] += metric_orig_train.base_rate(privileged = True)
    favor_metric_orig['priv_unfavor'] += 1 - metric_orig_train.base_rate(privileged = True)
    favor_metric_orig['unpriv_favor'] += metric_orig_train.base_rate(privileged = False)
    favor_metric_orig['unpriv_unfavor'] += 1 - metric_orig_train.base_rate(privileged = False)

    print(dataset_orig_train.features.shape, dataset_orig_val.features.shape, dataset_orig_test.features.shape)

    print('\n##########################################################\n')
    print('[INFO] Starting ' + BASELINE + ' Experiment ......')
    print('\n##########################################################\n\n')
    print('\n------------------------------\n')
    print('[INFO] ' + BASELINE + ' Original Results......')
    print('\n------------------------------\n')
    orig_metrics = orig_no_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, BASELINE, orig_metrics, SCALER)

    print('\n------------------------------\n')
    print('[INFO] LR Random Oversampling ......')
    print('\n------------------------------\n')
    metric_transf_train, transf_metrics = synth_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, unprivileged_groups, base_rate_privileged, base_rate_unprivileged, BASELINE, transf_metrics, f_label, uf_label, SCALER)

    # statistics of favored/positive class AFTER transf 
    favor_metric_transf['total_favor'] += metric_transf_train.base_rate()
    favor_metric_transf['total_unfavor'] += 1 - metric_transf_train.base_rate()
    favor_metric_transf['priv_favor'] += metric_transf_train.base_rate(privileged = True)
    favor_metric_transf['priv_unfavor'] += 1 - metric_transf_train.base_rate(privileged = True)
    favor_metric_transf['unpriv_favor'] += metric_transf_train.base_rate(privileged = False)
    favor_metric_transf['unpriv_unfavor'] += 1 - metric_transf_train.base_rate(privileged = False)

    if MITIGATOR == 'dir' or MITIGATOR == "all":
        # Disparat Impact Remover Mitigator
        print('\n------------------------------\n')
        print('[INFO] '+BASELINE+' preprocessing--disparat impact remover ......')
        print('\n------------------------------\n')
        dir_metrics = dir_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test,  sens_attr, BASELINE, dir_metrics, SCALER)

    if MITIGATOR == 'rew'  or MITIGATOR == "all":
        # Reweighing Mitigator 

        print('\n------------------------------\n')
        print('[INFO] ' +BASELINE+' preprocessing--reweighting ......')
        print('\n------------------------------\n')
        reweigh_metrics = reweigh_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test,  unprivileged_groups, privileged_groups, BASELINE, reweigh_metrics, SCALER)


    if MITIGATOR == 'egr'  or MITIGATOR == "all":
        # Exponentiation Gradient Reduction 

        print('\n------------------------------\n')
        print('\n[INFO:]'+BASELINE+' in-processing Exponentiation Gradient Reduction ...... \n')
        print('\n------------------------------\n')
        egr_metrics = egr_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, egr_metrics, BASELINE, SCALER)


    if BASELINE == 'lr' and (MITIGATOR == 'pr'  or MITIGATOR == "all"):
        # Prejudice Remover 
        print('\n------------------------------\n')
        print('\n[INFO:] '+BASELINE+' in-processing Prejudice Remover ...... \n')
        print('\n------------------------------\n')
        pr_metrics = pr_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, pr_metrics, SCALER)


    if MITIGATOR == 'cpp'  or MITIGATOR == "all":
        # CalibratedEqPostProcessing 
        print('\n------------------------------\n')
        print('\n[INFO:]'+BASELINE+' post-processing Calibrated Equal Odds ......\n')
        print('\n------------------------------\n')
        cpp_metrics = cpp_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, cpp_metrics, BASELINE, SCALER)


    if MITIGATOR == 'ro'  or MITIGATOR == "all":
        # RejectOptionClassification 
        print('\n------------------------------\n')
        print('\n[INFO:] '+BASELINE+' post-processing Reject Option Classification ......\n')
        print('\n------------------------------\n')
        ro_metrics = ro_mitigator(dataset_orig_train, dataset_orig_val, dataset_orig_test, ro_metrics, BASELINE, SCALER)

# dataframe to display favored  metrics

print('\n\n\n')
print(DATASET)
print(dataset_orig_train.features.shape[0])
print('\n\n\n')
priv_metric_orig = {k: [v/N] for (k,v) in priv_metric_orig.items()}
results = [priv_metric_orig]
tr = pd.Series(['orig'], name='num_instance')
df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis = 0).set_index([tr])
print(df)

print('\n')
favor_metric_orig = {k: [v/N] for (k,v) in favor_metric_orig.items()}
favor_metric_transf = {k: [v/N] for (k,v) in favor_metric_transf.items()}
pd.set_option('display.multi_sparse', False)
results = [favor_metric_orig, favor_metric_transf]
tr = pd.Series(['orig'] + ['transf'], name='dataset')
df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis = 0).set_index([tr])
print(df)

print('\n\n\n')

# dataframe to display fairness metrics
# error metrics
orig_error_metrics = {k: [statistics.stdev(v)] for (k,v) in orig_metrics.items()}
transf_error_metrics = {k: [statistics.stdev(v)] for (k,v) in transf_metrics.items()}
reweigh_error_metrics = {k: [statistics.stdev(v)] for (k,v) in reweigh_metrics.items()}
dir_error_metrics = {k: [statistics.stdev(v)] for (k,v) in dir_metrics.items()}
egr_error_metrics = {k: [statistics.stdev(v)] for (k,v) in egr_metrics.items()}
cpp_error_metrics = {k: [statistics.stdev(v)] for (k,v) in cpp_metrics.items()}
ro_error_metrics = {k: [statistics.stdev(v)] for (k,v) in ro_metrics.items()}
pr_error_metrics = {k: [statistics.stdev(v)] for (k,v) in pr_metrics.items()}

# mean value metrics
orig_metrics_mean = {k: [sum(v)/N] for (k,v) in orig_metrics.items()}
transf_metrics_mean = {k: [sum(v)/N] for (k,v) in transf_metrics.items()}
reweigh_metrics_mean = {k:[sum(v)/N] for (k,v) in reweigh_metrics.items()}
dir_metrics_mean = {k:[sum(v)/N] for (k,v) in dir_metrics.items()}
egr_metrics_mean = {k:[sum(v)/N] for (k,v) in egr_metrics.items()}
cpp_metrics_mean = {k: [sum(v)/N] for (k,v) in cpp_metrics.items()}
ro_metrics_mean = {k: [sum(v)/N] for (k,v) in ro_metrics.items()}
pr_metrics_mean = {k: [sum(v)/N] for (k,v) in pr_metrics.items()}

# Python paired sample t-test
from scipy.stats import ttest_rel
def paired_t (a, b):
    np_a = np.array(a)
    np_b = np.array(b)
    s, p = ttest_rel(np.absolute(np_a), np.absolute(np_b))
    return p

def acc_diff (a, b):
    np_a = np.array(a)
    np_b = np.array(b)
    delta = np_a - np_b
    m = statistics.mean(delta)
    s = statistics.stdev(delta)
    return [m, s]

pd.set_option('display.multi_sparse', False)

plt.rcParams.update({'font.size': 8}) # must set in top

results = [orig_metrics_mean,
        transf_metrics_mean,
        dir_metrics_mean,
        reweigh_metrics_mean,
        egr_metrics_mean,
        pr_metrics_mean,
        cpp_metrics_mean,
        ro_metrics_mean]

errors = [orig_error_metrics,
        transf_error_metrics,
        dir_error_metrics,
        reweigh_error_metrics,
        egr_error_metrics,
        pr_error_metrics,
        cpp_error_metrics,
        ro_error_metrics]

plot_baseline(results, errors, baseline=BASELINE.upper())
stat =  {k: [paired_t(transf_metrics[k], v)] for (k,v) in orig_metrics.items()}
print(stat)
plt.show()
