import partial_differential_functions as pdf
import stats_functions as stats
import vector_functions as vec

from sympy import *
import sympy as sym

import string

# setup for pretty print
init_printing()

letters = ' '.join(list(string.ascii_lowercase))
a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, theta = symbols(letters + ' theta')

pprint(theta)
print('\n')

pdf.line_spectrum(lambda w: 1/(2*pi**2) if w == 0 else 0, lambda w: cos(w*pi/2)/(pi**2*w**2) if w != 0 else 0)