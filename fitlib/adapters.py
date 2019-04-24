"""An *adapter* can transform the parameters passed to the fitting function
(workhorse) to a different form.

Since the fitting library (`scipy.optimize.curve_fit()`) only understands scalar
parameters, and the calculations might be written for a different data kind
(e.g. an object), the parameters must be converted to scalars (several scalars
per parameter are possible).

Such transformations are done via adapters. They have full control of
which parameters will be passed to the fitting functions and in which form.

The adapter receives complete information on the :meth:`FitHelper.fit()` call.
Three arguments to the constructor are:
    - `param_dict` is a dictionary of parameters supplied in some way
                    ( positional and keyword arguments of `fit()`,
                    arguments of the workhorse etc. ).
    - `params_bounds` is a dictionary with parameter names as keys and (lower, upper)
                    tuples as values. It should not be changed.
    - `fit_guess` is a list of guess values for `least_squares`. It should be
                    expanded with the initial values for parameters that
                    this adapter is responsible for.
    - `fit_bounds` is a tuple of two lists with bounds for `least_squares`.
                    It should be expanded with bounds for parameters that
                    this adapter is responsible for.

The adapter might keep references to these dictionaries and is expected to
mark some parameters in `param_dict` as its own (by setting the `adapter` field).

The adapter should support following methods:

    - `read_guess(guess, i)` :
        Convert fitting guess into parameter values in `param_dict`.
        This will be called with all i values in range(y.shape[0]).
    - `fill_coeff_dict(coeff_dict, guess, errors)` :
        Fill a dictionary with parameter values that will make it possible to
        reproduce a workhorse function call.

"""

from numpy import inf
from .parameters import DudParameter, ConstParameter, FitParameter

class EasyAdapter:
    """This adapter should cover most of `ParamAdapter` use cases while hiding
    a large part of the managing complexity.

    It has a list of predefined `FitParameter` fields that are each responsible
    for a single entry in `fit_kwargs`, `coeff_dict` and `fit_params` or `const_array_params`.
    Those keep track of their indices etc.

    When subclassing `EasyAdapter`, call :meth:`add_parameter` in the constructor
    for every fit parameter that you want to define. `EasyAdapter` takes care of
    only using parameters that are required to be fitted by the user. All the parameters
    that have been defined are available through `parameters` dictionary, while the
    names of parameters that are being fitted are available in `fitted_parameters` list.
    """

    def __init__(self,
                 param_dict,
                 params_bounds,
                 fit_guess,
                 fit_bounds,
                 ):
        self.param_dict = param_dict
        self.params_bounds = params_bounds
        self.__fh_guess = fit_guess
        self.__fh_bounds = fit_bounds
        self.parameters = {}

    def add_parameter(self, name, default_value):
        if name not in self.param_dict:
            # User doesn't care about this parameter.
            # I daresay we don't need it
            self.parameters[name] = DudParameter(name, default_value)
        else:
            p = self.param_dict[name]
            p.has_adapter = True

            if p.has_initial_value:
                value = p.initial_value
            else:
                value = default_value

            p.current_value = value

            if not p.fitted:
                # We just need to ensure those are translated correctly to WH args
                self.parameters[name] = ConstParameter(name, value, p.multiplicity)
            else:
                self.parameters[name] = \
                    FitParameter(name, value, p.multiplicity, len(self.__fh_guess))

                # Fill guess
                if p.multiplicity == 1:
                    self.__fh_guess.append(self.parameters[name].value)
                else:
                    self.__fh_guess.extend(self.parameters[name].value)

                # Fill bounds
                if name in self.params_bounds:
                    low, high = self.params_bounds[name]
                else:
                    low, high = -inf, inf

                for _ in range(p.multiplicity):
                    self.__fh_bounds[0].append(low)
                    self.__fh_bounds[1].append(high)

        return self.parameters[name]

    def read_guess(self, guess, i):
        for param in self.parameters.values():
            if param.fitted:
                param.read_value(guess)
            if param.name in self.param_dict:
                self.param_dict[param.name].current_value = param.value_at(i)

    def fill_coeff_dict(self, coeff_dict, params, errors):
        for param in self.parameters.values():
            if param.fitted:
                param.read_value(params)
                param.read_error(errors)

            coeff_name = param.name
            if param.multiplicity > 1:
                coeff_name += '[]'
            coeff_dict[coeff_name] = param.full_value
