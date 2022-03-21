"""
@brief: Plotting utilities module.

This module implements plotting functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lib_tds_rg.tds_rg_module import TheoreticalDistributionFitter
from statsmodels.distributions.empirical_distribution import ECDF


def tabulate_data(cols, headers, title=None):
    """
        @brief: Prints the given columns data as a table with headers and title.

        @param cols: The data inside the columns to be represented in the table.
        @param headers: The headers for each column.
        @param title: The title of the table (optional).
        """
    for i, col in enumerate(cols):
        cols[i] = [str(word) for word in col]
    col_lens = []
    for i, c in enumerate(cols):
        col_lens.append(max(map(len, [*c, headers[i]])))

    if title is not None:
        title_len = sum(col_lens)+3*len(col_lens)+1
        print(f'{title : ^{title_len}}')
    for l in col_lens:
        print('+', '-' * (l + 2), sep='', end='')
    print('+')
    for i, h in enumerate(headers):
        print(f'| {h : ^{col_lens[i] + 1}}', sep='', end='')
    print('|')
    for l in col_lens:
        print('+', '=' * (l + 2), sep='', end='')
    print('+')

    for word_index in range(len(cols[0])):
        for i, c in enumerate(cols):
            print(f'| {c[word_index] : <{col_lens[i] + 1}}', end='')
        print('|')
        for l in col_lens:
            print('+', '-' * (l + 2), sep='', end='')
        print('+')


class DistributionPlotter(object):
    """
    @brief: A class used for different distributions plotting.
    """

    @staticmethod
    def plot_kde(data, data_label=''):
        """
        @brief: Plots the kernel density estimation.

        @param data: The data from which the KDE evaluates.
        @param data_label: The label of the data (optional).
        """
        if len(data_label) > 0:
            data_label = '_' + data_label
        sns.kdeplot(sorted(data), label=f'kde{data_label}')

    @staticmethod
    def plot_histogram(data, data_label='', transparency_level=1):
        """
        @brief: Plots the histogram of the given data.

        @param data: The data from which the KDE evaluates.
        @param data_label: The label of the data (optional).
        @param transparency_level: The transparency level of the bars in the histogram.
        """
        if len(data_label) > 0:
            data_label = '_' + data_label
        plt.hist(sorted(data), density=True, label=f'hist{data_label}', alpha=transparency_level)

    @staticmethod
    def plot_theoretical_pdf(data, dist_name, plt_kwargs={}):
        """
        @brief: Plots the theoretical PDF of the data.

        @param data: The data from which the PDF is generated.
        @param dist_name: A theoretical distribution name, to which the data is fitted before plotted.
                          See @ref TheoreticalDistributionFitter.get_distribution_names.
        @param plt_kwargs: Additional parameters to the plotting function (optional).
        """
        sorted_data = sorted(data)
        pdf = TheoreticalDistributionFitter.get_fitted_pdf(data, dist_name)
        plt.plot(sorted_data, pdf(sorted_data), label=f'{dist_name}_pdf', **plt_kwargs)

    @staticmethod
    def plot_theoretical_cdf(data, dist_name, plt_kwargs={}):
        """
        @brief: Plots the theoretical CDF of the data.

        @param data: The data from which the CDF is generated.
        @param dist_name: A theoretical distribution name, to which the data is fitted before plotted.
                          See @ref TheoreticalDistributionFitter.get_distribution_names.
        @param plt_kwargs: Additional parameters to the plotting function (optional).
        """
        sorted_data = sorted(data)
        cdf = TheoreticalDistributionFitter.get_fitted_cdf(data, dist_name)
        plt.plot(sorted_data, cdf(sorted_data), label=f'{dist_name}_cdf', **plt_kwargs)

    @staticmethod
    def plot_histogram_and_kde(data, data_label='', transparency_level=1):
        """
        @brief: Plots the histogram and KDE of the data in the same graph.

        @param data: The data from which the KDE and histograms are generated.
        @param data_label: The label of the data (optional).
        @param transparency_level: The transparency level of the bars in the histogram.
        """
        DistributionPlotter.plot_histogram(data, data_label, transparency_level)
        DistributionPlotter.plot_kde(data, data_label)
        plt.legend()

    @staticmethod
    def plot_kde_vs_pdf(data, dist_name, data_label=''):
        """
        @brief: Plots the KDE and fitted PDF of the data in the same graph.

        @param data: The data from which the KDE and histograms are generated.
        @param dist_name: A theoretical distribution name, to which the data is fitted before plotted.
                          See @ref TheoreticalDistributionFitter.get_distribution_names.

        @param data_label: The label of the data (optional).
        """
        DistributionPlotter.plot_theoretical_pdf(dist_name)
        DistributionPlotter.plot_kde(data, data_label)
        plt.legend()

    @staticmethod
    def plot_ecdf(data, label=''):
        """
        @brief: Calculates and plots the empirical CDF for a data sample.

        @param data: Data to be plotted.
        @param label: Label to be plotted.
        """
        data_ecdf = ECDF(data)
        if len(label) > 0:
            label = '_' + label
        plt.plot(data_ecdf.x, data_ecdf.y, label=f'ECDF{label}')
        plt.ylabel('Cummulative Density')
        plt.legend()

    @staticmethod
    def plot_ecdfs(samples, labels='', title=''):
        """
        @brief: Calculates the empirical CDF for the given data samples and plot them on the same graph.

        @param samples: Iterable containing the samples to be plotted.
        @param labels: Data labels to be plotted.
        """
        if labels == '':
            labels = ['']*len(samples)
        for i, data in enumerate(samples):
            DistributionPlotter.plot_ecdf(data, label=labels[i])
        plt.legend()
        plt.title(title)

    @staticmethod
    def plot_kdes(samples):
        """
        @brief: Plots on the same graph the KDE of each data sample.

        @param samples: Iterable containing the samples to be plotted.
        """
        for data in samples:
            DistributionPlotter.plot_kde(data)
        plt.legend()

    @staticmethod
    def subplots_kdes_ecdfs(samples, figsize=(15, 5)):
        """
        @brief: Plots the KDEs and ECDFs of the data samples on different subplots.

        @param samples: Iterable containing the samples to be plotted.
        @param figsize: The figure size to be used.
        """
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        DistributionPlotter.plot_kdes(samples)
        plt.title("KDE")

        plt.subplot(1, 2, 2)
        DistributionPlotter.plot_ecdfs(samples)
        plt.title("ECDF")

    @staticmethod
    def plot_histograms_kdes(samples, labels=(), transparency_level=0.6, title_suffix=""):
        """
        @brief: Plots the histograms and KDEs of the data samples on the same graph.

        @param samples: Iterable containing the samples to be plotted.
        @param labels: Iterable containing the samples labels.
        @param transparency_level: The transparency level of the bars in the histograms (optional).
        @param title_suffix: A suffix to add to the title of the graph (optional).
        """
        if labels == ():
            labels = ['' for s in samples]
        plt.title("Histogram & KDE " + title_suffix)
        for i, data in enumerate(samples):
            DistributionPlotter.plot_histogram_and_kde(data, labels[i], transparency_level=transparency_level)

    @staticmethod
    def plot_top_k(data, top_results, show_cdf=True, cols=3):
        """
         @brief: Plots the top k theoretical distributions results.
                 See @ref TheoreticalDistributionFitter.top_k_theoretical_distributions.

         @param data: The data sample that was used.
         @param top_results: Tuple containing the distribution name and the p-value.
         @param cols: Amount of columns in the graph (optional).
         """
        # print a table with the results
        k = len(top_results)
        dist_names = [t[0] for t in top_results]
        p_values = [t[1] for t in top_results]
        tabulate_data(cols=[dist_names, p_values],
                      headers=['Distribution Name', 'P-Value'],
                      title=f'Top {k} theoretical distributions'
                      )
        # plot Histogram and KDE
        rows = int(np.ceil(k / cols))
        plt.figure(figsize=(cols * 7, rows * 5))
        for i, result in enumerate(top_results):
            dist_name = result[0]
            plt.subplot(rows, cols, i + 1)
            DistributionPlotter.plot_histogram_and_kde(data, data_label=dist_name)
            DistributionPlotter.plot_theoretical_pdf(data, dist_name, plt_kwargs={'c': 'k'})
            plt.legend()

        # plot CDF and ECDF
        if show_cdf is True:
            plt.figure(figsize=(cols * 7, rows * 5))
            for i, result in enumerate(top_results):
                dist_name = result[0]
                plt.subplot(rows, cols, i + 1)
                DistributionPlotter.plot_ecdf(data)
                DistributionPlotter.plot_theoretical_cdf(data, dist_name)
                plt.legend()
