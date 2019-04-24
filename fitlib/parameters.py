"""
.. autoclass ParameterKind
"""

import enum
import inspect
from numpy import asanyarray, inf

@enum.unique
class ParameterKind(enum.Enum):
    """Type of parameters to :meth:`FitHelper.fit()`
        - VIRTUAL : Must be controlled by an adapter
        - POSITIONAL : Should be in *args to workhorse
        - VAR_POSITIONAL : The workhorse accepts varargs
        - KEYWORD : Has to be supplied in **kwargs to workhorse
        - VAR_KEYWORD : The workhorse accepts varkwargs
    """
    VIRTUAL = enum.auto()
    POSITIONAL = enum.auto()
    VAR_POSITIONAL = enum.auto()
    KEYWORD = enum.auto()
    VAR_KEYWORD = enum.auto()

def convert_param_kind(k):
    """Convert :class:`inspect.Parameter.kind` to :class:`ParameterKind`"""
    if k == inspect.Parameter.POSITIONAL_ONLY or \
       k == inspect.Parameter.POSITIONAL_OR_KEYWORD:
        return ParameterKind.POSITIONAL
    if k == inspect.Parameter.VAR_POSITIONAL:
        return ParameterKind.VAR_POSITIONAL
    if k == inspect.Parameter.KEYWORD_ONLY:
        return ParameterKind.KEYWORD
    if k == inspect.Parameter.VAR_KEYWORD:
        return ParameterKind.VAR_KEYWORD
    return ParameterKind.VIRTUAL

class SuppliedParameter:
    """A parameter that is supplied in some way to :meth:`Fitter.fit()`"""

    def __init__(self,
                 name,
                 fitted=False,
                 has_adapter=False,
                 multiplicity=1,
                 has_initial_value=False,
                 initial_value=None,
                 kind=ParameterKind.VIRTUAL,
                 index=None):
        self.name = name
        self.fitted = fitted
        self.has_adapter = has_adapter
        self.multiplicity = multiplicity
        self._has_initial_value = has_initial_value
        self._initial_value = initial_value
        self.kind = kind
        self.index = index
        self.current_value = None

    def set_iv(self, value):
        "Set initial value"
        self._has_initial_value = True
        self._initial_value = value
        # self.current_value = value

    def del_iv(self):
        "Remove initial value"
        self._has_initial_value = False
        self._initial_value = None
        # self.current_value = None

    initial_value = property(
        lambda self: self._initial_value,
        set_iv, del_iv, "Initial value of a parameter")

    has_initial_value = property(
        lambda self: self._has_initial_value,
        doc="The parameter was supplied with an initial value")

    def __repr__(self):
        return f"SuppliedParameter({self.name}, {self.fitted}," \
               f" {self.has_adapter}, {self.multiplicity}," \
               f" {self.has_initial_value}, {self.initial_value}," \
               f" {self.kind}, {self.index})"

    def __str__(self):
        return f"SuppliedParameter({self.name}, {'fitted' if self.fitted else 'not fitted'}," \
               f" {'has adapter' if self.has_adapter else 'has no adapter'}, {self.multiplicity}," \
               f" {self.has_initial_value}, {self.initial_value}," \
               f" {self.kind}, {self.index})"

class _Parameter:
    """Blanket class for :class:`EasyAdapter` parameters"""
    def __init__(self, name, value, multiplicity, fitted):
        self.name = name

        self.value = value
        self.error = 0

        self.multiplicity = multiplicity
        self.fitted = fitted


    def read_value(self, array):
        """Get a value for this parameter from a list of fit variables."""
        pass

    def read_error(self, array):
        """Get an error for this parameter from a list of fit variables."""
        pass

    def value_at(self, i):
        """Get the i'th component of this value."""
        if self.multiplicity == 1:
            return self.value
        else:
            return self.value[i]

class DudParameter(_Parameter):
    """A parameter that the :class:`EasyAdapter` knows about,
    but the user doesn't mention in the call."""
    def __init__(self, name, value):
        super().__init__(name, value, 1, False)
        self.full_value = value

    def value_at(self, i):
        return self.value

    def __repr__(self):
        return f"DudParameter({self.name}, {self.value})"

class ConstParameter(_Parameter):
    """A parameter that is held constant during fit."""
    def __init__(self, name, value, multiplicity):
        super().__init__(name, value, multiplicity, False)
        self.full_value = value

        if multiplicity > 1:
            shape = asanyarray(self.value).shape
            if not shape:
                self.value = [self.value]*self.multiplicity
                self.full_value = self.value
            elif shape[0] != multiplicity:
                raise ValueError("Expected length of {} ({}) doesn't match "
                                 "initial value length ({})".format(self.name,
                                                                    multiplicity,
                                                                    shape[0] if shape else 0))

    def __repr__(self):
        return f"ConstParameter({self.name}, {self.value}, {self.multiplicity})"

class FitParameter(_Parameter):
    """A parameter that is varied during fit."""

    def __init__(self,
                 name, # Name of the parameter
                 value, # Initial value
                 multiplicity,
                 index
                ):
        super().__init__(name, value, multiplicity, True)
        self.error = [0]*multiplicity

        self.index = index
        self.slice = slice(index, index+multiplicity)
        self.bounds = (-inf, inf)

        if multiplicity > 1:
            shape = asanyarray(self.value).shape
            if not shape:
                self.value = [value]*multiplicity
            elif shape[0] != multiplicity:
                raise ValueError("Expected length of {} ({}) doesn't match "
                                 "initial value length ({})".format(self.name,
                                                                    multiplicity,
                                                                    shape[0]))

    def construct_full_value(self):
        """We only construct the value when called, since this is a rare operation"""
        if self.multiplicity == 1:
            return (self.value, self.error)
        else:
            return list(zip(self.value, self.error))

    full_value = property(construct_full_value,
                          doc="Value of the parameter including error. A list for multifits")

    def read_value(self, array):
        if self.multiplicity == 1:
            self.value = array[self.index]
        else:
            self.value = array[self.slice]

    def read_error(self, array):
        if self.multiplicity == 1:
            self.error = array[self.index]
        else:
            self.error = array[self.slice]

    def __str__(self):
        parts = []
        if not self.fitted:
            parts.append('const ')
        parts.append(self.name)
        if self.multiplicity > 1:
            parts.append('[]')
        parts.append(' ')
        parts.append(str(self.full_value))

        return ''.join(parts)

    def __repr__(self):
        return f"FitParameter({self.name!r}, {self.value}, {self.multiplicity}, {self.index})"
