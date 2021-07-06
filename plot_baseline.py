import pandas as pd
import statistics
import matplotlib.pyplot as plt

pd.set_option('display.multi_sparse', False)
plt.rcParams.update({'font.size': 8}) # must set in top

# plotting for a given baseline
#input: a list of fairness metrics

def plot_baseline(results, errors, baseline='LR'):
    if baseline == "LR":
        index = pd.Series([baseline+'_orig']+ [baseline+'_syn']+ [baseline+'_dir']+ [baseline+'_rew']+ [baseline+'_egr']+ [baseline+'_pr']+ [baseline+'_cpp']+ [baseline+'_ro'], name='Classifier Bias Mitigator')
    else:
        index = pd.Series([baseline+'_orig']+ [baseline+'_syn']+ [baseline+'_dir']+ [baseline+'_rew']+ [baseline+'_egr']+ [baseline+'_cpp']+ [baseline+'_ro'], name='Classifier Bias Mitigator')

    df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index(index)
    df_error = pd.concat([pd.DataFrame(metrics) for metrics in errors], axis=0).set_index(index)
    ax = df.plot.bar(yerr=df_error, capsize=4, rot=0, subplots=True, title=['', '', '', '', '', ''], fontsize = 12)
    if baseline == "LR":
        plt.savefig(baseline+'_syn_dir_rew_egr_pr_cpp_ro.eps', format='eps')
    else:
        plt.savefig(baseline+'_syn_dir_rew_egr_cpp_ro.eps', format='eps')
    print(df)

