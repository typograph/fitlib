"""fithelper - more comfortable fitting of arbitrary functions.

This module wraps `scipy.optimize.least_squares()` to allow for more freedom when
fitting of experimental XY data.

Consider the following function definition::

    def parabola(x, c, b, a):
        return a*x**2 + b*x + c

If we want to fit this function to some data `y_data` as function of `x_data`,
we usually call :meth:`scipy.optimize.curve_fit()` like this::

    >>> popt, pcov = curve_fit(parabola, x_data, y_data, [0,0,1])
    >>> print(popt)
    array([0.32, 0.55, 3.47])

The fitted values are now in `popt`, the covariance matrix in `pcov`. The fit is done.

The same fit in :mod:`fithelper` is performed like this::

    >>> fitter = FitHelper(parabola)
    >>> fit_params = fitter.fit(x_data, y_data, ['a', 'b', 'c'], a=1, b=0, c=0)
    >>> print(fit_params)
    {'a': (0.32, 0.01), 'b': (0.55, 0.1), 'c': (3.47, 1.01)}

Note that:

    1. You have to create a `fitter` object before fitting. It can be reused
       as long as you are fitting the same function, though.

    2. You can refer to the parameters of `parabola` by name rather than by position.
       This is usually more convenient, since you do not need to remember their order.
       You can, however, still refer to them by position::

        >>> fit_params = fitter.fit(x_data, y_data, ['a', 'b', 'c'], 0, 0, 1)
        >>> print(fit_params)
        {'a': (0.32, 0.01), 'b': (0.55, 0.1), 'c': (3.47, 1.01)}

    3. You have to specify explicitely which parameters to fit. If you do not wish to
       fit one or more of the parameters, you omit its name from the list::

        >>> fit_params = fitter.fit(x_data, y_data, ['b', 'c'], a=0.3, b=0, c=0)
        >>> print(fit_params)
        {'a': 0.3, 'b': (0.65, 0.1), 'c': (7.08, 0.83)}

    4. Instead of the covariance matrix, you get a standard deviation for every
       fitted variable.

Unlike :fun:`curve_fit()`, however, :mod:`fithelper` allows you to fit a larger variety
of functions and not only 1D data. Consider the function `peakpair`::


    Peak = namedtuple('Peak', ['A', 'w', 'x0'], defaults=[1,1,0])
    
    def gaussian(x, x0, A, w):
        return A*exp((x-x0)**2/w**2)

    def peakpair(energies, peak, kind, y0):
        if kind == 'symmetric':
            return gaussian(energies, peak.x0, peak.A, peak.w) + \
                   gaussian(energies, -peak.x0, peak.A, peak.w) + y0
        elif kind == 'antisymmetric':
            return gaussian(energies, peak.x0, peak.A, peak.w) + \
                   gaussian(energies, -peak.x0, -peak.A, peak.w) + y0
        else:
            raise ValueError(f"Unknown double peak kind '{kind}'")

This function computes a double peak structure, with either symmetric or antisymmetric peaks.
Note that the `peak` parameter is a (named) tuple and `kind` is a string. Neither can be 
fitted directly.
Now suppose that you have several spectra, some with symmetric, some with antisymmetric peaks,
but you know that all of them have the same (unknown) width. You can still fit all the data
simultaniously with just one `FitHelper.fit()` invocation. To do that, you have to add an
adapter that will split the `peak` object into individual components, supply a 2d array
of y values to `fit()` and specify which values should be fitted separately for array rows.

    >>> fitter = FitHelper(peakpair)
    >>> fitter.add_adapter_class(NamedTupleAdapter('peak', Peak))
    >>> fit_params = fitter.fit(x_data, np.stack((y_sym, y_asym)),
                                ['A[]', 'w'],
                                peak=Peak(x0='10.0'),
                                kind=['symmetric', 'antisymmetric'],
                                const_array_params=['kind'])
    >>> print(fit_params)
    {'A[]': [(0.32, 0.01), ()], 'x0': 10.0, 'peak': Peak(A=1.0, w=1.0, x0=10.0)}


The main features are:

   1. The fitting function can have an arbitrary high number of parameters, and only
   a subset of those will be fitted. The subset can be selected on each invocation
   of the fitting function, allowing to first fitting one parameter, then the second one
   etc.

   2. The Y data can have higher dimentionality than X data for fitting several curves
   simultaniously. At the time of the fit one can decide whether each parameter is
   the same for all curves or different. As before each parameter can be fixed or fitted.

"""

from .fithelper import FitHelper
