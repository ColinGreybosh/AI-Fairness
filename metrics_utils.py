# Metrics function
import numpy as np
from collections import OrderedDict, defaultdict
from aif360.metrics import ClassificationMetric

#setup classification/test models
from models import TModel


def test(f_label, uf_label, unprivileged_groups, privileged_groups, dataset, model, thresh_arr, metric_arrs):
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
        metric_arrs['acc'].append(metric.accuracy())
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
        metric_arrs['unpriv_fpr'].append(metric.false_positive_rate(privileged=False))
        metric_arrs['unpriv_fnr'].append(metric.false_negative_rate(privileged=False))
        metric_arrs['priv_fpr'].append(metric.false_positive_rate(privileged=True))
        metric_arrs['priv_fnr'].append(metric.false_negative_rate(privileged=True))
        metric_arrs['df_bias_amp'].append(metric.differential_fairness_bias_amplification())

    return metric_arrs


def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    metrics,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    if metrics is None:
        metrics = defaultdict(list)

        metrics['bal_acc'] = 0.5*(classified_metric_pred.true_positive_rate()+
                                                 classified_metric_pred.true_negative_rate())
        metrics['acc'] = classified_metric_pred.accuracy()
        metrics['avg_odds_diff'] = classified_metric_pred.average_odds_difference()
        metrics['disp_imp'] = classified_metric_pred.disparate_impact()
        metrics['stat_par_diff'] = classified_metric_pred.statistical_parity_difference()
        metrics['eq_opp_diff'] = classified_metric_pred.equal_opportunity_difference()
        metrics['theil_ind'] = classified_metric_pred.theil_index()
        metrics['df_bias_amp'] = classified_metric_pred.differential_fairness_bias_amplification()
    else:
        metrics['bal_acc'].append(0.5*(classified_metric_pred.true_positive_rate()+
                                                 classified_metric_pred.true_negative_rate()))
        metrics['acc'].append(classified_metric_pred.accuracy())
        metrics['avg_odds_diff'].append(classified_metric_pred.average_odds_difference()) 
        metrics['disp_imp'].append(classified_metric_pred.disparate_impact())
        metrics['stat_par_diff'].append(classified_metric_pred.statistical_parity_difference())
        metrics['eq_opp_diff'].append(classified_metric_pred.equal_opportunity_difference())
        metrics['theil_ind'].append(classified_metric_pred.theil_index())
        metrics['df_bias_amp'].append(classified_metric_pred.differential_fairness_bias_amplification())
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics


def describe_metrics(metrics, thresh_arr, TEST=True):
    if not TEST:
        best_ind = np.argmax(metrics['bal_acc'])
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    else:
        best_ind = -1
    print("Accuracy: {:6.4f}".format(metrics['acc'][-1]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    #disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = metrics['disp_imp'][best_ind]
    print("Corresponding disparate impact value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
    print("Corresponding DF bias amplification: {:6.4f}".format(metrics['df_bias_amp'][best_ind]))
#    print("Corresponding false positive_rate: {:6.4f}".format(metrics['false_positive_rate'][best_ind]))
#    print("Corresponding false negative_rate: {:6.4f}".format(metrics['false_negative_rate'][best_ind]))



def get_test_metrics(dataset_orig_train, dataset_orig_val, dataset_orig_test, model_type, test_metrics, f_label, uf_label, unprivileged_groups, privileged_groups, THRESH_ARR, DISPLAY, SCALER):

    dataset = dataset_orig_train

    test_model = TModel(model_type)
    mod_orig = test_model.set_model(dataset, SCALER)

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

    val_metrics = test(f_label, uf_label,
                       unprivileged_groups, privileged_groups,
                       dataset=dataset_orig_val_pred,
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

    #describe_metrics(val_metrics, thresh_arr)


    print('Testing Original ...')
    test_metrics = test(f_label, uf_label,
                        unprivileged_groups, privileged_groups,
                        dataset=dataset_orig_test_pred,
                        model=mod_orig,
                        thresh_arr=[thresh_arr[orig_best_ind]], metric_arrs=test_metrics)

    describe_metrics(test_metrics, thresh_arr)

    return test_metrics


