"""
"""

import numpy as np
from fithelper.fithelper import FitHelper
from fithelper.parameters import ParameterKind, SuppliedParameter
from fithelper.adapters import EasyAdapter
import matplotlib

def parabola(x, c, b, a):
    return a*x**2 + b*x + c

def polynomial(x, *coeffs):
    # print(coeffs)

    y = np.zeros_like(x)
    for c in reversed(coeffs):
        y *= x
        y += c
    return y

def random_parabola(x, c=None, b=None, a=None):
    if a is None:
        a = np.random.random()
    if b is None:
        b = np.random.random()
    if c is None:
        c = np.random.random()
    clean = parabola(x, c, b, a)
    scale = np.amax(clean) - np.amin(clean)
    return (c,b,a), clean + np.random.random(size=x.shape)*scale/10

class PolyAdapter(EasyAdapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.order = self.param_dict['order']
        self.order.has_adapter = True
        if not self.order.has_initial_value:
            raise ValueError("The order of polynomial to fit should be given")
        for i in range(self.order.initial_value+1):
            p_name = f'C{i}'
            sp_name = f'*coeffs_{i+1}'
            self.add_parameter(p_name, 0).alt_name = sp_name
            self.param_dict[sp_name] = \
                SuppliedParameter(sp_name,
                                  has_adapter=True,
                                  fitted=True,
                                  has_initial_value=True,
                                  initial_value=self.parameters[p_name].value_at(0),
                                  kind=ParameterKind.POSITIONAL,
                                  index=i)

    def read_guess(self, guess, i):
        super().read_guess(guess,i)
        for p in self.parameters.values():
            self.param_dict[p.alt_name].current_value = p.value_at(i)

if __name__ == "__main__":
    matplotlib.use('Qt5Agg') 

    # fitter_simple = FitHelper(parabola)
    # fitter_simple.set_bounds('c', 0, 1)

    x = np.linspace(-10,10,100)
    coeffs, y = random_parabola(x)

    # matplotlib.pyplot.figure()
    # fitter_simple.plot(x, y, a=1, b=0, c=0,
    #                    plot_kwargs=dict(linestyle='--'),
    #                    plot_ydata=False)

    # cs = fitter_simple.fit(x, y, ['a', 'b', 'c'], a=1, b=0, c=0)

    # fitter_simple.plot(x, y, coeff_dict=cs)


    # matplotlib.pyplot.show()

    y_multi = np.stack((y,
                        random_parabola(x, b=coeffs[-2])[1],
                        random_parabola(x, b=coeffs[-2])[1]))

    # matplotlib.pyplot.figure()
    # fitter_simple.plot(x, y_multi, a=1, b=0, c=0,
    #                    plot_kwargs=dict(linestyle='--'),
    #                    plot_ydata=False,
    #                    vertical_offset=20,
    #                    const_array_params=['a', 'c'])

    # cs = fitter_simple.fit(x, y_multi, ['a[]', 'b', 'c[]'], a=1, b=0, c=0)
    # fitter_simple.plot(x, y_multi, coeff_dict=cs, vertical_offset=20)

    # matplotlib.pyplot.show()


    # fitter_poly = FitHelper(polynomial)

    # matplotlib.pyplot.figure()
    # cs = fitter_poly.fit(x, y, ['*coeffs_1', '*coeffs_2', '*coeffs_3'], 0, 0, 0)
    # fitter_poly.plot(x, y, coeff_dict=cs)
    # matplotlib.pyplot.show()

    # matplotlib.pyplot.figure()
    # cs = fitter_poly.fit(x, y_multi, ['*coeffs_1[]', '*coeffs_2', '*coeffs_3[]'], 0, 0, 0)
    # fitter_poly.plot(x, y_multi, coeff_dict=cs)
    # matplotlib.pyplot.show()

    fitter_poly_adapt = FitHelper(polynomial)
    fitter_poly_adapt.add_adapter_class(PolyAdapter)

    matplotlib.pyplot.figure()
    cs = fitter_poly_adapt.fit(x, y, ['C0', 'C1', 'C2'], order=3)
    fitter_poly_adapt.plot(x, y, order=3, coeff_dict=cs)
    print(cs)
    matplotlib.pyplot.show()

    # fitter_polynomial = FitHelper(np.polynomial.polyval)
    # fitter_polynomial.add_adapter_class(PolynomialAdapter)

    # cs = fitter_polynomial.fit(x, y, ['C0', 'C1', 'C2'], order=3)
    # print(cs)
