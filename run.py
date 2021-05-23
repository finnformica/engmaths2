import partial_differential_functions as pdf
import stats_functions as stats
import vector_functions as vec

from sympy import *
import sympy as sym

import string

init_printing()

letters = ' '.join(list(string.ascii_lowercase))
a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, theta = symbols(letters + ' theta')

pprint(theta)
print('\n')

# function = [2*x**2 - 3*z, -2*x*y, -4*x]
# div = vec.divergence(function)
# pprint(vec.triple_integral(div, [z, y, x], [0, 4 - 2*x - 2*y], [0, 2 - x], [0, 2]))

# pprint(pdf.half_range_cosine_series_continuous(t, pi, t))

# Line Spectrum
# j = sqrt(-1)
# pdf.line_spectrum(lambda n: 0, lambda n: 2*sin(n) / (n*(j*n + 3)) if n != 0 else 0)

# stats.binomial_hypothesis(2, 0.05, 0.6, 6, 1)
#
# data1 = [19.1, 15.6, 5.2, 12.1, 9.1, 12.5]
# data2 = [19.2, 14.6, 5.2, 11.3, 7.3, 11.0]
# stats.paired_t_test(data1, data2)

observed = [10, 15, 25, 20, 25, 5]
expected = [15, 20, 15, 15, 20, 15]
print(stats.chisquared_test(observed, expected, 0.05, 2))
