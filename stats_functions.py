from scipy.stats import chisquare, chi2_contingency, chi2, poisson
from scipy import stats
from statistics import variance, stdev, mean, median
import math

from random import choices
from sympy import binomial, Eq, solve, symbols

a, b = symbols('a, b')


def p_value_to_z_score(p_value):
    z = stats.norm.ppf(p_value)
    return z


def p_value_to_t_score(p_value, sample_size):
    t = stats.t.ppf(p_value, sample_size)
    return t


def z_score_to_p_value(z_score):
    p = stats.norm.cdf(z_score)
    return p


def t_score_to_p_value(t_score, sample_size):
    p = stats.t.cdf(t_score, sample_size)
    return p


def sample_variance(data):  # tested
    mean_val = mean(data)
    nums = [(d - mean_val) ** 2 for d in data]
    s_squared = sum(nums) / (len(data) - 1)

    return s_squared


def coefficient_variation(args):
    return stdev(args) / mean(args)


def covariance(x, y):  # tested
    mean_x = mean(x)
    mean_y = mean(y)

    num = [(x[i] - mean_x) * (y[i] - mean_y) for i, _ in enumerate(x)]

    return sum(num) / (len(x) - 1)


def correlation_coefficient(list_x, list_y):  # tested

    sx = stdev(list_x)
    sy = stdev(list_y)

    return covariance(list_x, list_y) / (sx * sy)


def pearson_correlation_coefficient(list1, list2):
    x_mean = mean(list1)
    t_mean = mean(list2)

    num = sum([(list1[i] - x_mean)*(list2[i] - t_mean) for i, _ in enumerate(list1)])
    den1 = sum([(list1[i] - x_mean)**2 for i, _ in enumerate(list1)])
    den2 = sum([(list2[i] - t_mean)**2 for i, _ in enumerate(list1)])
    r = num / (math.sqrt(den1) * math.sqrt(den2))

    return r


def fisher_approximation(pearsons_coeff):  # tested
    return 0.5 * math.log((1 + pearsons_coeff) / (1 - pearsons_coeff))


def fisher_transformation(pearson_correlation_coeff, correlated, significance, sample_size):  # tested
    r = pearson_correlation_coeff
    rho = correlated
    statistic = 0.5 * (math.log((1 + r) / (1 - r)) - math.log((1 + rho) / (1 - rho))) * math.sqrt(sample_size - 3)
    z_score = p_value_to_z_score(1-significance / 2)
    if abs(statistic) < abs(z_score):
        print("Cannot reject null hypothesis")
    else:
        print("Reject null hypothesis")

    return statistic, z_score

    # example format
    # pearson_correlation_coeff = -1.30
    # rho = 0 | this suggests null hypothesis is that they are uncorrelated
    # significance = 0.10
    # sample_size = 5


def normalize_list(args):
    sd = stdev(args)
    return [(i - mean(args)) / sd for i in args]


def sample_error(args):
    sd = stdev(args)
    return sd / math.sqrt(len(args))


def get_confidence_interval_normal(sample_mean, confidence, sd, sample_size):
    alpha_val = 1 - confidence
    critical_probability = 1 - alpha_val / 2
    z_code = p_value_to_z_score(critical_probability)
    print("Z Code: {:.3f}".format(z_code))
    x = sample_mean - (z_code * (sd / math.sqrt(sample_size)))
    y = sample_mean + (z_code * (sd / math.sqrt(sample_size)))
    print("Confidence Interval:")
    print("Low value: {:.2f}".format(x))
    print("High value: {:.2f}".format(y))


def get_confidence_interval_t(sample_mean, confidence, sd, sample_size):
    alpha_val = 1 - confidence
    critical_probability = 1 - alpha_val / 2
    t_code = p_value_to_t_score(critical_probability, sample_size - 1)
    print("T Code: {:.3f}".format(t_code))
    x = sample_mean - (t_code * (sd / math.sqrt(sample_size)))
    y = sample_mean + (t_code * (sd / math.sqrt(sample_size)))
    print("Confidence Interval:")
    print("Low value: {:.2f}".format(x))
    print("High value: {:.2f}".format(y))


def get_z_score(sample, population_mean):
    sample_mean = mean(*sample)
    sd = stdev(*sample)
    sample_size = len(sample)

    return (sample_mean - population_mean) / (sd / math.sqrt(sample_size))


def t_critical_value(significance_value, dof, number_of_tails):  # tested
    significance_value = significance_value / number_of_tails
    critical_value = abs(stats.t.ppf(significance_value, dof))

    return critical_value


def get_t_score_from_data(sample, population_mean):
    sample_mean = mean(sample)
    sample_deviation = stdev(sample)
    sample_size = len(sample)

    return(sample_mean - population_mean) / (sample_deviation / math.sqrt(sample_size))


def t_statistic(sample_mean, population_mean, sample_deviation, sample_size):
    return (population_mean - sample_mean) / (sample_deviation / math.sqrt(sample_size))


def bootstrap_confidence_interval(sample, significance):
    xbar = [mean(choices(sample, k=10)) for i in range(0, 10000)]
    xbar.sort()
    divider = int(200/significance)
    confidence_interval = [xbar[10_000//divider], xbar[(10_000*(divider-1))//divider]]
    return confidence_interval


def chisquared_test(observed, expected, significance, dof):  # tested
    print('Chi squared test:')
    x = chisquare(observed, f_exp=expected)
    test_stat = x[0]
    pvalue = x[1]

    critical_value = chi2.ppf(1 - significance, dof)

    if test_stat > critical_value:
        print("Reject null hypothesis")
    else:
        print("Cannot reject null hypothesis")

    print(f'Test statistic = {test_stat}')
    print(f'Critical value = {critical_value}\n')

    return test_stat, pvalue


def chi2_critical_value(confidence, dof):
    chi2_critical = chi2.ppf(confidence, dof)
    return chi2_critical


def independence_test(significance_value, data):  # data takes two sets of values to test between.
    stat, p, dof, expected = chi2_contingency(data)
    # interpret p-value
    print("p value is " + str(p))
    critical = chi2_critical_value(1 - significance_value, dof)
    print("Critical value is {}. Test statisitic is {}. Therefore".format(critical, stat))
    if p <= significance_value:
        print('Dependent - reject null hypothesis')
    else:
        print('Independent - null hypothesis holds true')

    return stat, dof


def xbar(sample_data, groupings):

    total_freq = sum(sample_data)
    total_data = sum([sample_data[i] * groupings[i] for i, _ in enumerate(sample_data)])
    mu = total_data / total_freq

    return total_freq, mu

    # example
    # groupings = [0, 1, 2, 3, 4]
    # sample_data (observed) = [873, 77, 32, 16, 2]


def poisson_distribution(sample, groupings):
    print('Poisson distribution:')
    mu = xbar(sample, groupings)[1]
    total = xbar(sample, groupings)[0]
    expected = []
    for i in range(len(sample)):
        p = math.exp(-mu) * (mu ** i) / math.factorial(i)
        expected.append(round(p * total, 4))

    print(expected)
    return expected

    # example
    # observed = [873, 77, 32, 16, 2]
    # groupings = [0, 1, 2, 3, 4]


def normal_distribution(data, values_upper_bounds, mean, standard_deviation):
    print('Normal distribution:')
    expected = []
    prior_p = 0
    for i, value in enumerate(values_upper_bounds):
        z_score = (value - mean) / standard_deviation
        cumulative_p_value = z_score_to_p_value(z_score)

        if i == len(values_upper_bounds) - 1:
            p_value = 1 - prior_p
        else:
            p_value = cumulative_p_value - prior_p

        prior_p = cumulative_p_value
        expected.append(p_value * sum(data))

    print(expected)
    return expected

    # example format
    # observed data = [10, 32, 48, 10]
    # data_categories = [850, 900, 950, 1000]
    # s = math.sqrt(1625.3)
    # mean = 904


def binomial_distribution(probability_success, n_trials, x_outcomes):
    print('Binomial distrubution:')
    p = probability_success
    q = 1 - p
    n = n_trials
    x = x_outcomes
    probability = [binomial(n, i) * p**i * q**(n-i) for i in range(x + 1)]
    print(f'Pr(X <= {x_outcomes}) = {sum(probability)}\n')

    return probability


def binomial_hypothesis(no_tails, significance, probability_success, n_trials, x_outcomes):
    probability = sum(binomial_distribution(probability_success, n_trials, x_outcomes)) * no_tails
    print('Binomial hypothesis:')
    if probability <= significance:
        print('Reject the null hypothesis')
    else:
        print(f'Cannot reject the null hypothesis to {significance*100}% significance')
    print(f'Test statistic: {probability}')
    print(f'Critical value: {significance}\n')


def goodness_of_fit(observed, expected, significance, dof):
    print('Goodness-of-fit:')
    confidence = 1 - significance
    critical_value = chi2_critical_value(confidence, dof)
    statistic, pvalue = chisquared_test(observed, expected, significance, dof)

    print('Test statistic:', statistic)
    print('Critical value:', critical_value, '\n')


def statistically_significant(significance, sample1_size, sample1_mean, sample1_sd,
                              sample2_size, sample2_mean, sample2_sd, number_of_tails):
    global_deviation = math.sqrt(sample1_sd**2/sample1_size + sample2_sd**2/sample2_size)
    test_statistic = abs(sample1_mean - sample2_mean)/global_deviation
    critical_value = t_critical_value(significance, sample2_size - 1, number_of_tails)

    if test_statistic < critical_value:
        print('Cannot reject the null hypothesis')
    else:
        print('Reject the null hypothesis')

    print(f'Test statistic: {test_statistic}')
    print(f'Critical value: {critical_value}')


def residual(function, x_data, y_data):
    return [abs(function(x_data[i], y_data[i])) for i, _ in enumerate(x_data)]

    # example code
    # x_data = [1, 2, 3]
    # y_data = [20, 30, 70]
    # function = lambda x, y: 25*x - 10 - y


def find_expected_results(x, y):
    total = sum(x) + sum(y)
    row_tot = [sum(x), sum(y)]
    col_tot = [x[i] + y[i] for i, _ in enumerate(x)]
    out = [[col_tot[i] * row_tot[0]/total for i, _ in enumerate(x)],
           [col_tot[i] * row_tot[1]/total for i, _ in enumerate(y)]]

    print('Observed:', x + y)
    print('Expected:', out[0] + out[1], '\n')

    return out[0] + out[1]


def paired_t_test(data_1, data_2, significance, no_tails):
    print('Paired T test:')
    difference = [data_1[i] - data_2[i] for i, _ in enumerate(data_1)]

    x_bar = mean(difference)
    sd = math.sqrt(sample_variance(difference))

    test_stat = t_statistic(0, x_bar, sd, len(data_1))
    t_crit = t_critical_value(significance, len(data_1) - 1, no_tails)
    print(t_crit)

    if test_stat < t_crit:
        print("Accept null hypothesis. Mu = 0. No difference")
    else:
        print("Reject null hypothesis. Mu is different")

    print(f"Test statistic is: {test_stat}")
    print(f"Critical value is: {t_crit}\n")

    # This test assumes there is no difference between the data sets
    # mu is the mean differences between the sets of values
    # data1 = [140, 190, 50, 80]
    # data2 = [145, 192, 62, 87]


def linear_regression(x_data, y_data):
    x_i = sum(x_data)
    y_i = sum(y_data)
    x_squared = sum([x**2 for x in x_data])
    xy_i = sum([x_data[i] * y_data[i] for i, _ in enumerate(x_data)])
    n = len(x_data)

    eq1 = Eq(a * n + b * x_i, y_i)
    eq2 = Eq(a * x_i + b * x_squared, xy_i)
    regression_coeff = solve([eq1, eq2], (a, b))

    print('Regression line:')
    print(f'y = {regression_coeff[a]} + {regression_coeff[b]}x\n')
