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
mark some parameters in `param_dict` as its own (by setting the `has_adapter` field).

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

    def add_parameter(self, name, default_value, needs_initial_value=False):
        if name not in self.param_dict:
            # User doesn't care about this parameter.
            # I daresay we don't need it
            self.parameters[name] = DudParameter(name, default_value)
        else:
            p = self.param_dict[name]
            p.has_adapter = True

            if p.has_initial_value:
                value = p.initial_value
            elif needs_initial_value:
                raise ValueError(f'No initial value found for {name}.')
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


class BuildingAdapter(EasyAdapter):
    '''BuldingAdapter is a base class for adapters that need to make a set of objects
    fittable.'''
   
    def read_guess(self, guess, i=None):
        super().read_guess(guess, i)
        self.build_values(i)
            
    def get_parameter(self, name, needs_initial_value=True):
        if name not in self.param_dict:
            raise ValueError(f"{name} argument is needed for {self.__class__.__name__}"
                              " but it is not in the workhorse signature")

        if needs_initial_value and not self.param_dict[name].has_initial_value:
            raise ValueError(f"{name} argument is needed for {self.__class__.__name__}"
                              " but no value was supplied")
        
        return self.param_dict[name]
    

def NamedTupleAdapter(tuple_name, tuple_class):
    '''This function creates an adapter class for fitting named tuples of class `tuple_class`.
    `tuple_name` should be the name of the workhorse argument of this type.'''
    return type(f"NamedTupleAdapter_{tuple_class.__name__}_{tuple_name}",
                (NamedTupleAdapter_base,),
                {"tuple_parameter_name":tuple_name,
                 "tuple_class":tuple_class})



class NamedTupleAdapter_base(BuildingAdapter):
    tuple_class = None
    tuple_parameter_name = None

    def __new__(cls, *args):
        if cls == NamedTupleAdapter_base:
            raise ValueError("NamedTupleAdapter_base cannot be initialized directly")
        elif cls.tuple_class is None:
            raise ValueError("NamedTupleAdapter need to know the tuple's class to work")
        else:
            return object.__new__(cls)
    
    def __init__(self, *args):
        super().__init__(*args)
        
        self.tuple_obj = self.get_parameter(self.tuple_parameter_name, needs_initial_value=False)
        self.tuple_obj.fitted=True
        self.tuple_obj.has_adapter=True
        
        for field in self.tuple_class._fields:
            needs_initial_value = False
            if self.tuple_obj.has_initial_value:
                initial_value = self.tuple_obj.initial_value._asdict()[field]
            elif field in self.tuple_class._fields_defaults:
                initial_value = self.tuple_class._fields_defaults[field]
            else:
                initial_value = None
                needs_initial_value = True
            self.add_parameter(field, initial_value, needs_initial_value)

    def build_values(self, i):
        tuple_dict = {pname:p.value_at(i) for pname,p in self.parameters.items()}
        self.tuple_obj.current_value = self.tuple_class(**tuple_dict)            