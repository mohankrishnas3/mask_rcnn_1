# Copyright 2018-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import itertools
import scipy.optimize

_minus_inf = float('-inf')


def find_maximum(f, min_x, max_x, ftoll=1e-6, xtoll=1e-6, n=3, verbose=False):
    def binary_search(l, r, fl, fr):
        mid = l + (r - l) / 2
        fm = f(mid)
        binary_search.eval_count += 1
        if (abs(fm - fl) < ftoll and abs(fm - fr) < ftoll) or r - l < xtoll:
            return mid, fm
        if fl > fm >= fr:
            return binary_search(l, mid, fl, fm)
        if fl <= fm < fr:
            return binary_search(mid, r, fm, fr)
        p1, f1 = binary_search(l, mid, fl, fm)
        p2, f2 = binary_search(mid, r, fm, fr)
        if f1 > f2:
            return p1, f1
        else:
            return p2, f2

    def grid_binary_search(l, r, fl, fr):
        values = [fl]
        edges = [l]
        for i in range(n - 2):
            x = l + (i + 1) / n * (r - l)
            edges.append(x)
            values.append(f(x))
        values.append(fr)
        edges.append(r)
        index_max = max(range(len(values)), key=values.__getitem__)

        grid_binary_search.eval_count += n - 2

        if all(abs(x - values[index_max]) < ftoll for x in values) or r - l < xtoll:
            return edges[index_max], values[index_max]

        p1, f1 = None, _minus_inf
        p2, f2 = None, _minus_inf

        if index_max > 0:
            p1, f1 = grid_binary_search(edges[index_max - 1], edges[index_max], values[index_max - 1], values[index_max])
        if index_max < n - 1:
            p2, f2 = grid_binary_search(edges[index_max], edges[index_max + 1], values[index_max], values[index_max + 1])

        if f1 > f2:
            return p1, f1
        else:
            return p2, f2

    if n == 3:
        search_f = binary_search
    elif n > 3:
        search_f = grid_binary_search
    else:
        raise ValueError("n must be >= 3, got %d instead" % n)

    search_f.eval_count = 2
    best_th, best_value = search_f(min_x, max_x, f(min_x), f(max_x))
    if verbose:
        print("Found maximum %f at x = %f in %d evaluations" % (best_value, best_th, search_f.eval_count))
    return best_th, best_value


def find_maximum_mv(f, min_x, max_x, ftoll=1e-6, xtoll=1e-6, n=3, verbose=False):
    vf = np.vectorize(f, signature='(i)->()')

    if not n >= 3:
        raise ValueError("n must be >= 3, got %d instead" % n)

    min_x = np.asarray(min_x)
    max_x = np.asarray(max_x)

    if min_x.shape != max_x.shape:
        raise ValueError("min_x and max_x must be of the same shape, but got {} and {}".format(min_x.shape, max_x.shape))

    d = min_x.shape[0]

    cube = np.asarray(list(itertools.product(*zip([0] * d, [1] * d))))
    cube = np.reshape(cube, [2] * d + [d])
    linspace = np.linspace(0.0, 1.0, n)
    grid = np.stack(np.meshgrid(* [linspace] * d, indexing='ij'), axis=-1)

    min_x = np.reshape(min_x, [1] * d + [d])
    max_x = np.reshape(max_x, [1] * d + [d])

    def grid_nary_search(min_x, max_x):
        grid_this = min_x + grid * (max_x - min_x)
        values = vf(grid_this)
        grid_nary_search.eval_count += n ** d
        rargmax = np.argmax(values)
        iargmax = np.asarray(np.unravel_index(rargmax, values.shape))
        i_min = iargmax - 2
        i_max = iargmax + 2
        i_min = np.ravel_multi_index(i_min, grid_this.shape[:-1], mode='clip')
        i_max = np.ravel_multi_index(i_max, grid_this.shape[:-1], mode='clip')

        min_x, max_x = grid_this.reshape(-1, d)[i_min], grid_this.reshape(-1, d)[i_max]

        vmax = values.flatten()[rargmax]
        cmax = grid_this.reshape(-1, d)[rargmax]

        if all(abs(x - vmax) < ftoll for x in values.flatten()) or np.all(max_x - min_x < xtoll):
            return cmax, vmax

        return grid_nary_search(min_x, max_x)

    grid_nary_search.eval_count = 0

    best_th, best_value = grid_nary_search(min_x, max_x)
    if verbose:
        print("Found maximum {} at x = {} in {} evaluations".format(best_value, best_th, grid_nary_search.eval_count))
    return best_th, best_value


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    def function1(x):
        return (5 -(x + 3)**2 - 0.1 * (x - 3) ** 4) / 100 + math.cos(1.0 + x / 3.0) * 10.0 - x * 2

    def function2(x):
        x, y, z = x
        return 5.0 - (x - 2.0) ** 2 - (y - 1.0) ** 2 - (z + 1.0) ** 2

    #
    # mx, my = find_maximum(function1, -10, 10, verbose=True)
    #
    # mx, my = find_maximum(function1, -10, 10, n=100, verbose=True)
    #
    # mx, my = find_maximum_mv(function2, [-10, -10, -10], [10, 10.0, 10.0], n=6, verbose=True)
    #

    def rosen(x):
        """The Rosenbrock function"""
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    #
    # mx, my = find_maximum_mv(rosen, [-10, -10, -10, -10, -10], [10.0, 10.0, 10.0, 10.0, 10.0], n=6, verbose=True)

    r = scipy.optimize.minimize(rosen, [-10, -10, -10, -10, -10], method='Nelder-Mead', options={
        'disp': True,
        'initial_simplex': None,
        'maxiter': None,
        'xatol': 0.0001,
        # 'return_all': False,
        'fatol': 0.0001,
        # 'func': None,
        'maxfev': 100000
    })
    print(r)

    # x = np.arange(-10, 10, 0.2)
    #
    # plt.plot(x, np.vectorize(function1)(x))
    #
    # plt.annotate('max', xy=(mx, my), xytext=(mx + 1, my + 0.5),
    #              arrowprops=dict(facecolor='black', shrink=0.05),
    #              )
    #
    # plt.show()
