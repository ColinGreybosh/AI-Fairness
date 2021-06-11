# Metrics function
from collections import OrderedDict, defaultdict
from aif360.metrics import ClassificationMetric

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
        metrics['avg_odds_diff'] = classified_metric_pred.average_odds_difference()
        metrics['disp_imp'] = 1-min(classified_metric_pred.disparate_impact(), 1/classified_metric_pred.disparate_impact())
        metrics['stat_par_diff'] = classified_metric_pred.statistical_parity_difference()
        metrics['eq_opp_diff'] = classified_metric_pred.equal_opportunity_difference()
        metrics['theil_ind'] = classified_metric_pred.theil_index()
    else:
        metrics['bal_acc'].append(0.5*(classified_metric_pred.true_positive_rate()+
                                                 classified_metric_pred.true_negative_rate()))
        metrics['avg_odds_diff'].append(classified_metric_pred.average_odds_difference()) 
        metrics['disp_imp'].append(1-min(classified_metric_pred.disparate_impact(), 1/classified_metric_pred.disparate_impact()))
        metrics['stat_par_diff'].append(classified_metric_pred.statistical_parity_difference())
        metrics['eq_opp_diff'].append(classified_metric_pred.equal_opportunity_difference())
        metrics['theil_ind'].append(classified_metric_pred.theil_index())
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics


