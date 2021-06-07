import math
import sympy as sym
from numpy import linalg as la
import numpy as np
from sympy.vector import ParametricRegion, vector_integrate
from sympy import *
init_printing()

x, y, z, i, j, k, t, u, v, a, b, c, r, theta = sym.symbols('x y z i j k t u v a b c r theta')


def magnitude(vector):
    return sym.sqrt(sum([i**2 for i in vector]))


def dot_product(v1, v2):
    if len(v1) == len(v2):
        return sum([v1[i]*v2[i] for i, _ in enumerate(v1)])
    else:
        print("Vector dimensions must be consistent.")


def cross_product(vector1, vector2):
    if len(vector1) == 3 and len(vector2) == 3:
        i = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
        j = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
        k = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])
        return [i, j, k]
    else:
        print("Vector must have 3 dimensions.")


def triple_scalar_product(vector1, vector2, vector3):  # finds volume of parallelepiped
    i = vector3[0] * ((vector1[1] * vector2[2]) - (vector1[2] * vector2[1]))
    j = vector3[1] * ((vector1[2] * vector2[0]) - (vector1[0] * vector2[2]))
    k = vector3[2] * ((vector1[0] * vector2[1]) - (vector1[1] * vector2[0]))
    return abs(i + j + k)


def find_angle(vector1, vector2):  # angle between vectors
    dot_prod = dot_product(vector1, vector2)
    cos_theta = dot_prod / (magnitude(vector1) * magnitude(vector2))
    degrees = 180 * math.acos(cos_theta) / math.pi
    return round(degrees, 3)


def eigenvector(matrix):  # matrix in the form [[a, b],
    #                                           [c, d]]
    array = np.array(matrix)
    eigenvalues, eigenvectors = la.eig(array)
    set1 = (eigenvalues[0], [eigenvectors[0][0], eigenvectors[1][0]])
    print("Eigenvalue is: " + str(set1[0]) + ". Corresponding unit eigenvector is: " + str(set1[1]))
    set2 = (eigenvalues[1], [eigenvectors[0][1], eigenvectors[1][1]])
    print("Eigenvalue is: " + str(set2[0]) + ". Corresponding unit eigenvector is: " + str(set2[1]))
    return set1, set2, eigenvalues, eigenvectors


def two_dimensional_gradient(function):  # returns grad(f) of a scalar field
    diff_x = sym.diff(function, x)
    diff_y = sym.diff(function, y)
    gradient = [diff_x, diff_y]
    return gradient


def three_dimensional_gradient(function):
    diff_x = sym.diff(function, x)
    diff_y = sym.diff(function, y)
    diff_z = sym.diff(function, z)
    gradient = [diff_x, diff_y, diff_z]
    pprint(gradient)
    return gradient


# def grad(function):
#     if len(function) == 2:
#         diff_x = sym.diff(function, x)
#         diff_y = sym.diff(function, y)
#         gradient = [diff_x, diff_y]
#         return gradient
#
#     else:
#         diff_x = sym.diff(function, x)
#         diff_y = sym.diff(function, y)
#         diff_z = sym.diff(function, z)
#         gradient = [diff_x, diff_y, diff_z]
#         return gradient


def directional_derivative(function, point, direction):
    if len(point) == 2:
        grad = two_dimensional_gradient(function)
        mag = magnitude(direction)
        dot = dot_product(direction, grad) / mag
        sub_x = dot.subs(x, point[0])
        result = sub_x.subs(y, point[1])
        return result

    elif len(point) == 3:
        grad = three_dimensional_gradient(function)
        mag = magnitude(direction)
        dot = dot_product(direction, grad) / mag
        sub_x = dot.subs(x, point[0])
        sub_y = sub_x.subs(y, point[1])
        derivative = sub_y.subs(z, point[2])
        return derivative


def gradient_of_scalar(scalar, point):  # finds gradient / direction of scalar field
    grad = []
    if len(point) == 2:
        gradient = two_dimensional_gradient(scalar)
        for variable in gradient:
            grad_x = variable.subs(x, point[0])
            grad_y = grad_x.subs(y, point[1])
            grad.append(grad_y)

    elif len(point) == 3:
        gradient = three_dimensional_gradient(scalar)
        for variable in gradient:
            grad_x = variable.subs(x, point[0])
            grad_y = grad_x.subs(y, point[1])
            grad_z = grad_y.subs(z, point[2])
            grad.append(grad_z)
    return grad


def tangent_plane_of_surface(surface, point):  # returns answer in format [[x, y, z], scalar]
    grad = gradient_of_scalar(surface, point)
    scalar = dot_product(grad, point)
    return grad, scalar


def stationary_values_2variables(function):
    f_x = two_dimensional_gradient(function)[0]
    f_y = two_dimensional_gradient(function)[1]

    answer = sym.solve([f_x, f_y], (x, y))

    f_xx = two_dimensional_gradient(f_x)[0]
    f_xy = two_dimensional_gradient(f_x)[1]
    f_yx = two_dimensional_gradient(f_y)[0]
    f_yy = two_dimensional_gradient(f_y)[1]
    hessian = [[f_xx, f_xy], [f_yx, f_yy]]

    variables = []
    if type(answer) is dict:
        soltuions = []
        for key, value in answer.items():
            soltuions.append(value)
        answer = soltuions
        variables.append(answer)
    else:
        variables = answer

    for solution in variables:
        solved_hessian = []
        for row in hessian:
            new_row =[]
            for column in row:
                hessian_x = column.subs(x, solution[0])
                hessian_xy = hessian_x.subs(y, solution[1])
                new_row.append(hessian_xy)
            solved_hessian.append(new_row)
        array = np.array(solved_hessian, dtype=np.float32)
        eigenvalues, eigenvectors = la.eig(array)
        if eigenvalues[0] * eigenvalues[1] > 0 and eigenvalues[0] + eigenvalues[1] > 0:
            print('Stationary point {}, has the eigenvalues {}, therefore is a minimum'.format(solution, eigenvalues))

        elif eigenvalues[0] * eigenvalues[1] > 0 and eigenvalues[0] + eigenvalues[1] < 0:
            print('Stationary point {}, has the eigenvalues {}, therefore is a maximum'.format(solution, eigenvalues))

        elif eigenvalues[0] * eigenvalues[1] < 0:
            print('Stationary point {}, has the eigenvalues {}, therefore is a saddle point'.format(solution, eigenvalues))


def stationary_values_3variables(function):
    f_x = three_dimensional_gradient(function)[0]
    f_y = three_dimensional_gradient(function)[1]
    f_z = three_dimensional_gradient(function)[2]

    answer = sym.solve([f_x, f_y, f_z], (x, y, z))

    f_xx = three_dimensional_gradient(f_x)[0]
    f_xy = three_dimensional_gradient(f_x)[1]
    f_xz = three_dimensional_gradient(f_x)[2]

    f_yx = three_dimensional_gradient(f_y)[0]
    f_yy = three_dimensional_gradient(f_y)[1]
    f_yz = three_dimensional_gradient(f_y)[2]

    f_zx = three_dimensional_gradient(f_z)[0]
    f_zy = three_dimensional_gradient(f_z)[1]
    f_zz = three_dimensional_gradient(f_z)[2]
    hessian = [[f_xx, f_xy, f_xz], [f_yx, f_yy, f_yz], [f_zx, f_zy, f_zz]]

    variables = []
    if type(answer) is dict:
        soltuions = []
        for key, value in answer.items():
            soltuions.append(value)
        answer = soltuions
        variables.append(answer)
    else:
        variables = answer

    for solution in variables:
        solved_hessian = []
        for row in hessian:
            new_row =[]
            for column in row:
                hessian_x = column.subs(x, solution[0])
                hessian_xy = hessian_x.subs(y, solution[1])
                hessian_xyz = hessian_xy.subs(z, solution[2])
                new_row.append(hessian_xyz)
            solved_hessian.append(new_row)
        try:
            array = np.array(solved_hessian, dtype=np.float32)
        except TypeError:
            continue
        eigenvalues, eigenvectors = la.eig(array)

        if np.sign(eigenvalues[0]) == 1 and np.sign(eigenvalues[1]) == 1 and np.sign(eigenvalues[2]) == 1:
            print('Stationary point {}, has the eigenvalues {}, therefore is a minimum'.format(solution, eigenvalues))

        elif np.sign(eigenvalues[0]) == -1 and np.sign(eigenvalues[1]) == -1 and np.sign(eigenvalues[2]) == -1:
            print('Stationary point {}, has the eigenvalues {}, therefore is a maximum'.format(solution, eigenvalues))

        else:
            print('Stationary point {}, has the eigenvalues {}, therefore is a saddle point'.format(solution,
                                                                                                    eigenvalues))


def divergence(vector_field):  # all maths functions passed through must use sym. instead of math.
    var_x = sym.diff(vector_field[0], x)
    var_y = sym.diff(vector_field[1], y)
    var_z = sym.diff(vector_field[2], z)
    div = var_x + var_y + var_z
    print('Divergence:', div, '\n')
    return div


def curl(vector_field):  # all maths functions passed through must use sym. instead of math.
    i = (sym.diff(vector_field[2], y) - sym.diff(vector_field[1], z)).simplify()
    j = (sym.diff(vector_field[0], z) - sym.diff(vector_field[2], x)).simplify()
    k = (sym.diff(vector_field[1], x) - sym.diff(vector_field[0], y)).simplify()
    curl = [i, j, k]
    print('Curl:', curl, '\n')
    return curl


def arc_length(function, lower_bound, upper_bound):
    mag_squared = 0
    for term in function:
        mag_squared += (sym.diff(term, t))**2
    mag_rdash = sym.sqrt(mag_squared)
    s = sym.Integral(mag_rdash, (t, lower_bound, upper_bound)).evalf()
    return s


def scalar_path_integral(vector_field, parametrised_function, bounds):
    mag_squared = 0
    for term in parametrised_function:
        mag_squared += (sym.diff(term, t))**2
    mag_diff_r = sym.sqrt(mag_squared)
    integralx = vector_field.subs(x, parametrised_function[0])
    integralxy = integralx.subs(y, parametrised_function[1])
    integralxyz = integralxy.subs(z, parametrised_function[2])
    s = sym.Integral(integralxyz * mag_diff_r, (t, bounds[0], bounds[1])).evalf()
    return s
    # example input
    # vector_field = x**2 + y**2
    # param = [t, 2*t, 0]
    # bounds = [0, 1]


def work_integral(vector_field, parametrised_function, bounds):
    diff_r = []
    for term in parametrised_function:
        diff_r.append(sym.diff(term, t))
    work = dot_product(diff_r, vector_field)
    integralx = work.subs(x, parametrised_function[0])
    integralxy = integralx.subs(y, parametrised_function[1])
    integralxyz = integralxy.subs(z, parametrised_function[2])
    s = sym.Integral(integralxyz, (t, bounds[0], bounds[1])).evalf()
    print('Work integral:')
    pprint(s)
    return s


def double_integral(function, variable_1, bounds_1, variable_2, bounds_2):
    return sym.integrate(function, (variable_1, bounds_1[0], bounds_1[1]), (variable_2, bounds_2[0], bounds_2[1]))
    # example formatting
    # function = y**2
    # x_bounds = [y**2, y]
    # y_bounds = [0, 1]


def triple_integral(function, order_of_xyz, first_bounds, second_bounds, third_bounds):
    return sym.integrate(function, (order_of_xyz[0], first_bounds[0], first_bounds[1]), (order_of_xyz[1], second_bounds[0], second_bounds[1]), (order_of_xyz[2], third_bounds[0], third_bounds[1]))
    # example formatting
    # function = sym.sqrt(x**2 + z**2) * x
    # order_of_xyz = [x,y,z]
    # first_bounds = [0, z]
    # second_bounds = [0, sym.pi * 2]
    # third_bounds = [0, 3]


def parametrised_normal(surface, direction=1):
    du = [sym.diff(surface[i], u) for i, _ in enumerate(surface)]
    dv = [sym.diff(surface[i], v) * direction for i, _ in enumerate(surface)]
    return cross_product(du, dv)


def partial_differential_vector(terms, variable):
    return [sym.diff(dim, variable) for dim in terms]


def flux_integral(vector, surface, u_bounds, v_bounds):
    r_u = partial_differential_vector(surface, u)
    r_v = partial_differential_vector(surface, v)

    vector_parametrised = []
    for variable in vector:
        vector_parametrised.append(variable.subs({x: surface[0], y: surface[1], z: surface[2]}))

    dA = cross_product(r_u, r_v)
    integral = dot_product(vector, dA)
    flux = double_integral(integral, u, u_bounds, v, v_bounds)
    return flux

    # example formatting
    # vector = [3*u**2, v**2, 0]
    # surface = [u, v, 2*u + 3*v]
    # u_bounds = [0, 2]
    # v_bounds = [-1, 1]


def scalar_surface_integral(vector_field, surface, u_bounds, v_bounds):
    region = ParametricRegion((surface[0], surface[1], surface[2]), (u, u_bounds[0], u_bounds[1]), (v, v_bounds[0], v_bounds[1]))
    surface_integral = vector_integrate(vector_field, region)
    return surface_integral
    # example formatting
    # surface = [3*cos(v)*sin(u), 3*sin(v)*sin(u), 3*cos(u)]
    # vector_field = 3*cos(u)
    # u_bounds = [0, pi/2]
    # v_bounds = [0, 2*pi]


def stokes(line_c, vector_field, u_bounds, v_bounds):
    curl_xyz = curl(vector_field)
    curl_parametrised = []
    for variable in curl_xyz:
        curl_parametrised.append(variable.subs({x: line_c[0], y: line_c[1], z: line_c[0]}))
    r_u = partial_differential_vector(line_c, u)
    r_v = partial_differential_vector(line_c, v)

    dA = cross_product(r_u, r_v)
    integral = dot_product(curl_parametrised, dA)

    solution = double_integral(integral, u, u_bounds, v, v_bounds)
    return solution


def gauss(vector_field, r_bounds, u_bounds, v_bounds, parameterised):
    div = divergence(vector_field)
    subs = div.subs({x: parameterised[0], y: parameterised[1], z: parameterised[2]})
    simplified = sym.simplify(subs)
    changed = jacobian(parameterised, [r, u, v])
    integral = simplified * changed
    print(integral)
    ans = sym.integrate(integral, (r, r_bounds[0], r_bounds[1]), (u, u_bounds[0], u_bounds[1]),
                        (v, v_bounds[0], v_bounds[1]))
    return ans
    # example : this is for spherical polar co-ordinates
    # r_bounds = [0, 1]
    # u_bounds = [0, pi]
    # v_bounds = [0, 2*pi]
    # vector_field = [x*y**2, y*cos(z)+y*z**2, x**2*z-sin(z)]
    # parameterised = [r * sin(u) * cos(v), r * sin(u) * sin(v), r * cos(u)]
    # If integral is in x, y, z give param as : [r, u, v]


def determinant(mat):
    if len(mat) == 2:
        c = mat[0][0]*mat[1][1]-mat[1][0]*mat[0][1]
        return c
    elif len(mat) == 3:
        i = mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1]
        j = mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0]
        k = mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0]
        c = mat[0][0]*i - mat[0][1]*j + mat[0][2]*k
        return c


def jacobian(term, variables):
    jacob = []
    for i in range(3):
        new_row = []
        differential = term[i]
        for j in range(3):
            value = sym.diff(differential, variables[j])
            new_row.append(value)
        jacob.append(new_row)

    det = determinant(jacob)
    det_1 = sym.simplify(det)
    return det_1
