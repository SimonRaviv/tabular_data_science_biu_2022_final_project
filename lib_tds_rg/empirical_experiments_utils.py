"""
@brief This module supports the empirical distribution comparisons.

It uses the @ref tds_rg_module module for calculations, statistical tests and plotting utilities.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from lib_tds_rg.tds_rg_module import StatisticalTests as stest
from lib_tds_rg.tds_rg_module import RGSimilarity as rg_sim
from lib_tds_rg.plotting_utils import DistributionPlotter as plotter
from lib_tds_rg.plotting_utils import tabulate_data


def print_scores(rg_score, ks_pvalue, cvm_pvalue):
    rg_score_percent = round((1 - rg_score) * 100, 4)
    tabulate_data(cols=[['RG score', 'KS P-Value', 'CVM P-Value'],
                        [str(rg_score_percent) + ' [%]', ks_pvalue, cvm_pvalue]],
                  headers=['Metric', 'Value'])


def get_empirical_comparison(dist_sample1, dist_sample2, scale_data=None):
    rg_score = rg_sim.rg_empirical_similarity(sample1=dist_sample1, sample2=dist_sample2, scale_data=scale_data)
    ks_pvalue = stest.ks_2_sample_test(dist_sample1, dist_sample2)
    cvm_pvalue = stest.cvm_2_sample_test(dist_sample1, dist_sample2)

    return (rg_score, ks_pvalue, cvm_pvalue)


def print_experiment_scores(rg_score, ks_pvalue, cvm_pvalue, title_suffix=""):
    addition = ""
    if title_suffix != "":
        addition += f" with {title_suffix}"

    rg_score_percent = round((1 - rg_score) * 100, 4)
    tabulate_data(cols=[['RG score', 'KS P-Value', 'CVM P-Value'],
                        [str(rg_score_percent) + ' [%]', ks_pvalue, cvm_pvalue]],
                  headers=['Metric', 'Value'],
                  title=f'Comparison Scores{addition}')


def show_2_sample_helper(data1, data2, samples_labels="", title_suffix="", scale_data=False):
    rg_score, ks_pvalue, cvm_pvalue = get_empirical_comparison(data1, data2, scale_data=scale_data)
    print_experiment_scores(rg_score, ks_pvalue, cvm_pvalue, title_suffix=title_suffix)
    if scale_data is True:
        data1, data2 = rg_sim.get_scaled_data(data1, data2)
    if type(samples_labels) is str:
        labels = [samples_labels]*2
    else:
        labels = samples_labels
    plotter.plot_histograms_kdes([data1, data2], labels=labels, title_suffix=title_suffix)


def show_2_sample_comparisons(data1, data2, samples_labels="", title_suffix="", scale_data=False, show_ecdfs=False):
    plt.figure(figsize=(7*2, 5))
    plt.subplot(1, 2, 1)
    show_2_sample_helper(data1, data2, samples_labels=samples_labels, title_suffix=title_suffix)
    if scale_data is True:
        plt.subplot(1, 2, 2)
        show_2_sample_helper(data1, data2, samples_labels=samples_labels,
                             title_suffix=title_suffix+" Scaled data", scale_data=True)

    if show_ecdfs is True:
        ecdf_title = "ECDF" + title_suffix
        plt.figure(figsize=(7*2, 5))
        plt.subplot(1, 2, 1)
        plotter.plot_ecdfs([data1, data2], samples_labels, ecdf_title)
        if scale_data is True:
            plt.subplot(1, 2, 2)
            data1, data2 = rg_sim.get_scaled_data(data1, data2)
            plotter.plot_ecdfs([data1, data2], samples_labels, ecdf_title+" Scaled data")


def log_multi_test_statistics(
        rg_scores, ks_pvalues, cvm_pvalues,
        scores_units=(),
        plot_titles=("KS-Pvalues", "CVM-Pvalues", "RG-Similarity Percentage")):
    values = [np.mean(rg_scores), np.std(rg_scores),
              np.min(rg_scores), np.max(rg_scores),
              np.mean(ks_pvalues), np.std(ks_pvalues), np.min(ks_pvalues),
              np.max(ks_pvalues), np.mean(cvm_pvalues), np.std(cvm_pvalues),
              np.min(cvm_pvalues), np.max(cvm_pvalues)]

    for i, unit in enumerate(scores_units):
        values[i] = str(values[i]) + ' ' + str(unit)

    stats = ["rg_scores_mean", "rg_scores_std", "rg_scores_min",
             "rg_scores_max", "ks_pvalues_mean", "ks_pvalues_std",
             "ks_pvalues_min", "ks_pvalues_max", "cvm_pvalues_mean",
             "cvm_pvalues_std", "cvm_pvalues_min", "cvm_pvalues_max"]
    tabulate_data(cols=[stats, values],
                  headers=['Statistics Metric', 'Value'],
                  title='Results statistics')
    xs = list(range(1, len(rg_scores)+1))
    plt.figure(figsize=(30, 7))
    plt.subplot(1, 3, 1)
    sns.barplot(xs, ks_pvalues).set_title(plot_titles[0])
    plt.subplot(1, 3, 2)
    sns.barplot(xs, cvm_pvalues).set_title(plot_titles[1])
    plt.subplot(1, 3, 3)
    sns.barplot(xs, rg_scores).set_title(plot_titles[2])
