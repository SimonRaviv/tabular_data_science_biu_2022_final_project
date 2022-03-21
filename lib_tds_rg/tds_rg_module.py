"""
@brief: Tabular Data Science final project main module.

This module implements a metric for distributions similarities.
"""
import numpy as np

from scipy import stats as st
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale


class StatisticalTests(object):
    """
    @brief: A class for different statistical tests.
    """

    @staticmethod
    def ks_test(data, cdf):
        """
        @brief: Performs a Kolmogorov-Smirnov test between empirical data and a theoretical CDF.

        @param data: The empirical data to be used in the KS test.
        @param cdf: The theoretical CDF to be compared in the KS test.

        @return: The p-value result of the KS test.
        """
        res = st.kstest(data, cdf)
        return res.pvalue

    @staticmethod
    def ks_2_sample_test(data1, data2):
        """
        @brief: Performs a 2 sample Kolmogorov-Smirnov test between two empirical features.

        @param data1: The first empirical data to be used in the KS test.
        @param data2: The second empirical data to be used in the KS test.

        @return: the p-value result of the KS test.
        """
        res = st.ks_2samp(data1, data2)
        return res.pvalue

    @staticmethod
    def cvm_test(data, cdf):
        """
        @brief: Performs a Cramer Von Mises test between empirical data and a theoretical CDF.

        @param data: The empirical data to be used in the CVM test.
        @param cdf: The theoretical CDF to be compared in the CVM test.

        @return: The p-value result of the CVM test.
        """
        res = st.cramervonmises(data, cdf)
        return res.pvalue

    @staticmethod
    def cvm_2_sample_test(data1, data2):
        """
        @brief: Performs a 2 sample Cramer Von Mises test between two empirical features.

        @param data1: The first empirical data to be used in the CVM test.
        @param data2: The second empirical data to be used in the CVM test.

        @return: The p-value result of the CVM test.
        """
        res = st.cramervonmises_2samp(data1, data2)
        return res.pvalue


class TheoreticalDistributionFitter(object):
    """
    @brief: A class used to represent a theoretical distribution fitter for a given data sample.
    """

    @staticmethod
    def get_distribution_names():
        """
        @brief: A list of all the valid continuous distribution names.

        @return: Return the valid continuous distribution names.
        """
        continuous_distn_names = st._continuous_distns._distn_names
        ignored_distributions = \
            ['erlang', 'frechet_r', 'frechet_l', 'levy_stable',
             'mielke', 'kstwo', 'argus', 'studentized_range', 'nakagami']
        return [name for name in continuous_distn_names if name not in ignored_distributions]

    @staticmethod
    def get_fitted_pdf(data, dist_name):
        """
        @brief: Fits the distribution parameters to a given theoretical distribution computes.

        @param data: The data being used to fit the distribution.
        @param dist_name: A theoretical distribution name, from which the data is fitted.
                          See @ref TheoreticalDistributionFitter.get_distribution_names.

        @return: Return the PDF function fitted to the data.
        """
        # Get distribution generator
        distribution = getattr(st, dist_name)
        # Fit the parameters of the distribution
        params = distribution.fit(sorted(data))
        # Separate parts of parameters
        args = params[:-2]
        loc_param = params[-2]
        scale_param = params[-1]

        return lambda x: distribution.pdf(x, loc=loc_param, scale=scale_param, *args)

    @staticmethod
    def get_fitted_cdf(data, dist_name):
        """
        @brief: Fits the distribution parameters to a given theoretical distribution computes.

        @param data: The data being used to fit the distribution.
        @param dist_name: A theoretical distribution name, from which the data is fitted.
                          See @ref TheoreticalDistributionFitter.get_distribution_names.

        @return: Return the CDF function fitted to the data.
        """
        # Get distribution generator
        distribution = getattr(st, dist_name)
        # Fit the parameters of the distribution
        params = distribution.fit(sorted(data))
        # Separate parts of parameters
        args = params[:-2]
        loc_param = params[-2]
        scale_param = params[-1]

        return lambda x: distribution.cdf(x, loc=loc_param, scale=scale_param, *args)

    @staticmethod
    def top_k_theoretical_distributions(data, k=3):
        """
        @brief: Calculates the top K theoretical continuous distributions.

        This comes to serve as a view of what are the top K theoretical distribution the data could be sampled from.
        The theoretical CDF built by fitting the parameters using maximum likelihood estimation (MLE).
        The scoring metric based on combined P-Values of Kolmogorov-Smirnov and Cramer-von Mises statistical tests
        using the Fisher method.

        @param data: The data compared to the theoretical distributions.
        @param k: The number of top results to return.

        @return: The top k p-values and their corresponding theoretical distribution names.
        """
        results = []
        data = sorted(data)
        results = []
        statistical_tests_fn = [StatisticalTests.ks_test, StatisticalTests.cvm_test]
        distribution_names = TheoreticalDistributionFitter.get_distribution_names()
        for distribution in distribution_names:
            try:
                dist_cdf = TheoreticalDistributionFitter.get_fitted_cdf(data, distribution)
                combined_p_value = [stats_test_fn(data, dist_cdf) for stats_test_fn in statistical_tests_fn]
                _, combined_p_value = st.combine_pvalues(combined_p_value)
                if combined_p_value != float('nan'):
                    results.append((distribution, combined_p_value))
            except Exception:
                print(f'Failed to fit {distribution} distribution')
        # sort the results based on p-value in descending order
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:k]


class RGSimilarity(object):
    """
    @brief: A class used to calculate the similarity score based on the Raviv-Gavriely method.
    """

    @staticmethod
    def get_scaled_data(sample1, sample2):
        """
        @brief: Uses a min-max scale on the data samples.

        @return: Scaled data samples in range [0,1].
        """
        scaled_data1 = minmax_scale(sample1, feature_range=(0, 1))
        scaled_data2 = minmax_scale(sample2, feature_range=(0, 1))
        return scaled_data1, scaled_data2

    @staticmethod
    def rg_empirical_similarity(sample1, sample2, scale_data=True, kde_estimator_func=st.gaussian_kde):
        """
        @brief: Calculates the similarity score based on the RG method.

        RG method:
        1. Scale the samples so they are on the same domain (optional)
        2. Calculate the KDE for each sample
        3. Evaluate the values of the KDEs on each of the data sample points
        4. Compute a root mean squared error between the KDEs values
        5. Normalize the result to range [0,1]

        @param scale_data: Control the scaling process in the RG method.
        if true, the data is scaled to the same domain [0,1].
        @param kde_estimator_func: A callable used to generate the KDE function (default is a gaussian kde).
                                   An example can be seen in @ref scipy.stats.gaussian_kde.

        @return: A score in range [0,1] that represent the similarity between the two data distributions.
        """
        # scale the samples so they are on the same domain
        if scale_data is True:
            sample1, sample2 = RGSimilarity.get_scaled_data(sample1, sample2)

        # calculate the KDE for each sample
        feature1_kde = kde_estimator_func(sample1.reshape(-1, ))
        feature2_kde = kde_estimator_func(sample2.reshape(-1, ))

        unique_xs = np.unique(np.concatenate([sample1, sample2]))
        unique_xs.sort()

        # evaluate the values of the KDEs on each of the data sample points
        feature1_kde_values = feature1_kde(unique_xs)
        feature2_kde_values = feature2_kde(unique_xs)

        # calculate and normalize the RMSE score to range [0,1]
        rmse = mean_squared_error(feature1_kde_values, feature2_kde_values, squared=False)
        min_element = min(feature1_kde_values.min(), feature2_kde_values.min())
        max_element = max(feature1_kde_values.max(), feature2_kde_values.max())
        normalized_rmse = rmse / (max_element - min_element)

        return normalized_rmse
