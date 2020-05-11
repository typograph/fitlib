""":class:`FitHelper` is defined here."""

import inspect
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from numpy import inf, sqrt, diag, zeros_like, finfo, amax, asanyarray, zeros, prod
from numpy.linalg import svd

#from collections import namedtuple

from .adapters import EasyAdapter
from .parameters import SuppliedParameter, ParameterKind, convert_param_kind

class DefaultAdapter(EasyAdapter):
    """This is a `ParamAdapter` that is used internally after all other adapters
    have executed.

    It can only fit simple numeric parameters.
    """
    def __init__(self, *args):
        super().__init__(*args)

        for p in self.param_dict.values():
            # if (p.fitted or p.multiplicity > 1) and not p.has_adapter:
            if p.kind == ParameterKind.VAR_POSITIONAL:
                continue
            if p.kind == ParameterKind.VAR_KEYWORD:
                continue

            if not p.has_adapter:
                if p.kind == ParameterKind.VIRTUAL:
                    raise ValueError(f"There is no parameter called {p.name}.")
                if not p.has_initial_value:
                    raise ValueError(f"No initial value supplied for {p.name}")
                self.add_parameter(p.name, None)
                p.has_adapter = True

class FitHelper:
    """This class helps fitting arbitrary functions while varying
    some of the parameters and keeping others constant.

    To create a new fitter (you only need one fitter per function to fit),
    initialize an object of this class with the fitting function (*workhorse*)
    as the sole parameter.

    There is no limit on the signature of the workhorse. It is recommended
    to place constant arguments first and fit parameters second, but this is
    not a requirement.

    If the workhorse uses non-scalar parameters (lists, objects or similar)
    use a `ParamAdapter` to switch between parameter representations.

    To use the fitter, call the `fit()` function with appropriate parameters.
    To plot the result, call the `plot()` function using the result of `fit()`.
    """
    def __init__(self, workhorse):
        self.workhorse = workhorse
        self.workhorse_signature = inspect.signature(workhorse)

        self.initial_param_dict = {}
        for i, p in enumerate(self.workhorse_signature.parameters.values()):
            if i == 0: # We skip the first argument, since this is the X axis
                continue
            self.initial_param_dict[p.name] = \
                SuppliedParameter(
                    name=p.name,
                    has_initial_value=p.default is not inspect.Parameter.empty,
                    initial_value=p.default,
                    kind=convert_param_kind(p.kind),
                    index=i-1,
                )

        self.bounds = defaultdict(lambda: (-inf, inf))
        self.adapters = []

    def set_bounds(self, parameter, lower, upper):
        """In the case some parameters to the workhorse have to be constrained
        during the fit, add an appropriate bound using this method. `parameter`
        should be the name of the respective parameter (either defined by an
        adapter, or present in the workhorse signature), `upper` and `lower`
        are the respective bounds.
        Use `-np.inf` and `np.inf` if there are no constraints.
        """
        if lower == -inf and upper == inf:
            if parameter in self.bounds:
                del self.bounds[parameter]
        else:
            self.bounds[parameter] = (lower, upper)

    def add_adapter_class(self, adapter_class):
        """An adapter reformats the list of arguments for the workhorse, exposing
        internal parameters of complex objects. The adapters are instantiated
        and called in the order they are added.

        Note that a new adapter is created every time the `fit()` function is called.
        """
        self.adapters.append(adapter_class)

    def fill_param_dict(self, args, kwargs,
                              fit_params, const_array_params,
                              coeff_dict, multi_fit_N):
        """Fill a dict based on the `fit` call arguments. For every parameter
        try to determine initial value and multifit options.
        """
        
        param_dict = {k:SuppliedParameter(name=p.name,
                                          has_initial_value=p.has_initial_value,
                                          initial_value=p.initial_value,
                                          kind=p.kind,
                                          multiplicity=None,
                                          index=p.index)
                      for k, p in self.initial_param_dict.items()}

        ## Populate from args:
        
        for i, wh_arg in enumerate(args):

            # Find a parameter with this index
            params = [p for p in param_dict.values() if p.index == i]
            if params:
                p = params[0]
                if p.kind == ParameterKind.POSITIONAL:
                    p.initial_value = wh_arg
                elif p.kind == ParameterKind.VAR_POSITIONAL:
                    # user is suppying additional args. From here on all parameters
                    # will be positional and are nameless. The rest are keywords
                    for j in range(i, len(args)):
                        pnj = "*{}_{:d}".format(p.name, j-i+1)
                        if pnj in param_dict:
                            raise ValueError("How did you manage to have"
                                             "an argument named {} in your function?".format(pnj))
                        param_dict[pnj] = SuppliedParameter(
                            name=pnj,
                            kind=ParameterKind.POSITIONAL,
                            index=j,
                            multiplicity=None # Will be set later
                            )
                        param_dict[pnj].initial_value = args[j]
                    break
                else: # We should probably not accept keyword-only parameters as positional
                    raise ValueError("The {} function"
                                     "only accepts {} positional arguments,"
                                     "but {} were supplied".format(
                                         self.workhorse.__name__,
                                         i, len(args)))
            else:
                raise ValueError("The {} function"
                                 "only accepts {} arguments,"
                                 "but {} were supplied".format(
                                     self.workhorse.__name__,
                                     i, len(args)))

        ## Populate from coeff_dict

        def unwrap_pm_value(value):
            if isinstance(value, tuple):
                return value[0]
            else:
                return value

        p_varargs = [p for p in param_dict.values() if p.kind == ParameterKind.VAR_POSITIONAL]
        if p_varargs:
            p_varargs = p_varargs[0]
        else:
            p_varargs = None

        for pname in coeff_dict:

            multivalue = pname.endswith('[]')

            # Coeff_dict has the lowest priority.
            # We should ignore values that have been given more explicitely

            if multivalue:
                value = list(map(unwrap_pm_value, coeff_dict[pname]))
                pname = pname[:-2]
            else:
                value = unwrap_pm_value(coeff_dict[pname])

            if pname in param_dict:
                p = param_dict[pname]
                if p.multiplicity is None:
                    if multivalue:
                        if len(value) != multi_fit_N:
                            raise ValueError("Cannot use an array parameter"
                                             f" of size {len(value)}"
                                             f" for {pname}: y data has {multi_fit_N} sets")
                        p.multiplicity = multi_fit_N
                    else:
                        p.multiplicity = 1
                    p.initial_value = value
                elif p.multiplicity == 1:
                    if multivalue:
                        raise ValueError("Cannot use an array parameter"
                                         f" of size {len(value)}"
                                         f" for {pname}: it has been declared as shared")
                    p.initial_value = value
                elif not multivalue:
                    p.initial_value = [value]*p.multiplicity
                elif p.multiplicity != len(value):
                    raise ValueError("Cannot use an array parameter"
                                     f" of size {len(value)}"
                                     f" for {pname} of declared size {p.multiplicity}")
                else:
                    p.initial_value = value

            else:
                if multivalue:
                    if len(value) != multi_fit_N:
                        raise ValueError("Cannot use an array parameter"
                                         f" of size {len(value)}"
                                         f" for {pname}: y data has {multi_fit_N} sets")
                    p_multiplicity = multi_fit_N
                else:
                    p_multiplicity = 1

                if pname.startswith('*'):
                    # This must be a varargs component
                    if p_varargs is None:
                        raise ValueError(f"The {self.workhorse.__name__} function"
                                         " does not accept arbitrary arguments,"
                                         f" yet {pname} is given as a parameter")
                    if not pname[1:].startswith(p_varargs.name):
                        raise ValueError(f"The {self.workhorse.__name__} function"
                                         " vararg parameter is called {p_varargs.name},"
                                         f" yet {pname} is given as a parameter")
                    param_dict[pname] = \
                        SuppliedParameter(pname,
                                          multiplicity=p_multiplicity,
                                          kind=ParameterKind.POSITIONAL,
                                          index=int(pname[len(p_varargs.name)+2:])-1)
                else:
                    param_dict[pname] = SuppliedParameter(pname, multiplicity=p_multiplicity)
                param_dict[pname].initial_value = value
                
        ## Populate from kwargs

        for wh_kwarg in kwargs:
            if wh_kwarg not in param_dict:
                param_dict[wh_kwarg] = SuppliedParameter(wh_kwarg)
            param_dict[wh_kwarg].initial_value = kwargs[wh_kwarg]

        ## Populate from fit_params:

        for pname in fit_params:
            if pname.endswith('[]'):
                pname = pname[:-2]
                if multi_fit_N > 1:
                    p_multiplicity = multi_fit_N
                else:
                    raise ValueError("Cannot use an array parameter"
                                     " for {}: y data is 1D".format(pname))
            else:
                p_multiplicity = 1

            if pname in param_dict:
                param_dict[pname].fitted = True
                param_dict[pname].multiplicity = p_multiplicity
            else:
                param_dict[pname] = \
                    SuppliedParameter(
                        name=pname,
                        fitted=True,
                        multiplicity=p_multiplicity,
                    )

        ## Populate from const_array_params:

        for pname in const_array_params:
            if pname.endswith('[]'):
                pname = pname[:-2]

            if pname in param_dict:
                if param_dict[pname].fitted:
                    raise ValueError("{} cannot be a constant"
                                     " and be fitted at the same time".format(pname))
                param_dict[pname].multiplicity = multi_fit_N
            else:
                param_dict[pname] = \
                    SuppliedParameter(
                        name=pname,
                        multiplicity=multi_fit_N,
                    )

        # Everything that has multiplicity None should have 1 instead

        for p in param_dict.values():
            if p.multiplicity is None:
                p.multiplicity = 1

        # if p_varargs is not None:
            # del param_dict[p_varargs.name]

        return param_dict

    def fit(self,
            xdata, # 1D array for x axis
            ydata, # 1 or 2D-array for y axis. The x axis should be the last axis
            fit_params, # Names of parameters to fit. Add "[]" to parameter name to get a list
            *args, # Any positional parameters to pass to the calculating function
            const_array_params=(), # Names of parameters to not fit that are arrays.
        #    plot_initial=False, # Plot initial guess
        #    plot_kwargs={}, # Additional plotting parameters
            lstsq_kwargs={}, # Additional arguments to least_squares
            coeff_dict={}, # Parameters from an earlier run
            **kwargs # Parameters to fit or not fit
            ):
        """Uses `scipy.optimize.least_squares()` to fit the workhorse to the data.

        This function takes an arbitrary number of arguments according to the following:

        `xdata` is a 1D array of x-axis values (shape `(m,)`).
        `ydata` is a 1D or 2D array of data values (shape `(n, m)`).
        `fit_params` is a list of parameter names that should be fitted. These should be
          the same names that the workhorse uses. In the case that `ydata` is a 2D array,
          '[]' can be appended to a name to indicate that this parameter should be fitted
          individually for each curve.

        The parameters above should be passed as arguments, not as keyword arguments.
        They are followed by an arbitrary number of parameters that are passed directly
        to the workhorse without change.

        The rest of the parameters are keyword parameters. Every keyword parameter that
        matches a workhorse parameter and is in `fit_params` will be fitted. Its value will
        be used as an initial value. In the case of individual fitting of a parameter to various curves,
        the initial value should be a list of appropriate length. Workhorse parameters that are not in
        `fit_params` will have constant values. Every argument of the workhorse should
        be supplied with an initial value, either directly in the `fit()` call, or
        as a default value in the workhorse definition. Supplying a keyword argument that is
        not a workhorse argument, and is not understood by an adapter is a runtime error.

        `fit()` also takes the following keyword arguments:

        `const_array_params` is a list of parameters that are constant arrays, that is they
         are not fitted but should be initialized to different values for different curves.
        `lstsq_kwargs` is a dictionary of keyword parameteres for `scipy.optimize.least_squares()`.
        `coeff_dict` is a dictionary returned by a previous run of the `fit()` function. The values
          in the dictionary will be used as initial values, unless they were specified in a different form.

        `fit()` returns a dictionary of workhorse arguments, including both fitted and constant
          parameters.

        """

        fit_params = list(fit_params)
        const_array_params = list(const_array_params)

        ## This part decides wherether X and Y are in good shape

        y_shape = ydata.shape
        if len(y_shape) > 2:
            raise ValueError("Cannot fit {}-dimensional data (yet)".format(len(y_shape)))
        elif len(xdata) != y_shape[-1]:
            raise ValueError("X and Y lengths do not match")
        elif len(y_shape) == 2:
            multi_fit = True
            multi_fit_N = y_shape[0]
        else:
            multi_fit = False
            multi_fit_N = 1

        if const_array_params and not multi_fit:
            raise ValueError("Cannot use array parameters: y data is 1D")

        # First we put together all the parameters we now know about
        param_dict = self.fill_param_dict(args, kwargs,
                                          fit_params, const_array_params,
                                          coeff_dict, multi_fit_N)

        ## Initialize adapters for this run
        #
        # Every adapter goes through param_dict and sets `has_adapter`
        # for parameters it wants to control.
        #

        param_adapters = []
        fit_vars = []
        fit_bounds = ([], [])

        for adapter_class in self.adapters:
            param_adapters.append(adapter_class(param_dict, self.bounds, fit_vars, fit_bounds))

        # The default adapter takes care of what's left.

        param_adapters.append(DefaultAdapter(param_dict, self.bounds, fit_vars, fit_bounds))

        ## wh_args is a list of arguments to workhorse(*wh_args)
        # It contains constant parameters and initial values
        # for fitted parameters
        # In the multifit case, wh_args is a tuple of lists.

        wh_args = [None for p in param_dict.values() if p.kind == ParameterKind.POSITIONAL]
        wh_kwargs = {} # {p.name:None for p in param_dict.values() if p.kind == ParameterKind.KEYWORD}

        def fill_wh_args(args, kwargs, pdict):
            for p in pdict.values():
                if p.kind == ParameterKind.POSITIONAL:
                    args[p.index] = p.current_value
                elif p.kind == ParameterKind.KEYWORD:
                    kwargs[p.name] = p.current_value
                    
        fill_wh_args(wh_args, wh_kwargs, param_dict)

        ## least_squares takes a function of form fun(p, *args, **kwargs)
        # that returns a 1D array, and minimizes sum of squares
        #
        # We need to coerse the workhorse to this form
        # flattening Y if multifitting.
        # The helper function uses adapters to convert between
        # the flat parameter array supplied by least_squares
        # and the argument list for workhorse.

        def helper(x_data, p_values):
            p_values = list(p_values)
            for adapter in param_adapters:
                adapter.read_guesses(p_values)

            def single_guess(i):
                for adapter in param_adapters:
                    adapter.read_single_guess(p_values, i)
                fill_wh_args(wh_args, wh_kwargs, param_dict)
                return self.workhorse(x_data, *wh_args, **wh_kwargs)          
                
            if multi_fit:
                y_data = zeros_like(ydata)
                for i in range(multi_fit_N):
                    y_data[i] = single_guess(i)
                return y_data
            else:
                return single_guess(0)

        ## Do the actual fit.
        # This uses 'lm' method of least_squares for bounded
        # and 'trf' for unbounded problems.

        if 'method' in lstsq_kwargs:
            lstsq_method = lstsq_kwargs.pop('method')
        elif self.bounds:
            lstsq_method = 'trf'
        else:
            lstsq_method = 'lm'

        result = least_squares(lambda p_values : (helper(xdata, p_values) - ydata).flatten(),
                               fit_vars,
                               bounds=fit_bounds,
                               method=lstsq_method,
                               **lstsq_kwargs)
        if not result.success:
            raise RuntimeError("Fit failed: " + result.message)
        else:
            _, s, VT = svd(result.jac, full_matrices=False)
            threshold = finfo(float).eps * amax(result.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[:s.size]
            pcov = (VT.T / s**2) @ VT
            # res.cost is half sum of squares!
            pcov = pcov * 2 * result.cost / (len(xdata)*multi_fit_N - len(result.x))

            coeff_dict = {}
            for adapter in param_adapters:
                adapter.fill_coeff_dict(coeff_dict, result.x, sqrt(diag(pcov)))

        return coeff_dict

    @classmethod
    def plot_waterfall(cls, xdata, ydata, vertical_offset=0,
                       colors=None, labels=None, **kwargs):
        """Plot all pairs in `ydata`/`xdata` with a shift of `vertical_offset`.

        If you supply `colors`, it should be a list of length equal to
        number of expected curves. All curves and markers will be colored
        accordingly.

        Any additional keywords will be passed to Axes.plot and would take
        precedence over `colors`.
        """

        xdata = asanyarray(xdata)
        ydata = asanyarray(ydata)

        ncurves = ydata.shape[0] if len(ydata.shape) > len(xdata.shape) else 1

        if labels is None:
            labels = ['']*ncurves

        if colors is None:
            all_colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
            more_colors = all_colors*(ncurves // len(all_colors) + 1)
            colors = more_colors[:ncurves]
        elif len(colors) != ncurves:
            raise ValueError("Number of supplied colors does not correspond to number of curves")

        # Line colors
        if 'color' in kwargs:
            line_colors = [kwargs['color']]*ncurves
            del kwargs['color']
        else:
            line_colors = colors

        # Marker edge colors
        if 'markeredgecolor' in kwargs:
            edge_colors = [kwargs['markeredgecolor']]*ncurves
            del kwargs['markeredgecolor']
        else:
            edge_colors = colors

        if 'markerfacecolor' in kwargs:
            face_colors = [kwargs['markerfacecolor']]*ncurves
            del kwargs['markerfacecolor']
        else:
            face_colors = colors

        if len(ydata.shape) > len(xdata.shape): # For the case y.shape = (1, **)
            for i in range(ncurves):
                plt.plot(xdata, ydata[i] + vertical_offset*i,
                         color=line_colors[i],
                         markeredgecolor=edge_colors[i],
                         markerfacecolor=face_colors[i],
                         label=labels[i],
                         **kwargs)
        else:
            plt.plot(xdata, ydata,
                     color=line_colors[0],
                     markeredgecolor=edge_colors[0],
                     markerfacecolor=face_colors[0],
                     label=labels[0],
                     **kwargs)

        return colors

    def _plot(self,
              xdata, # 1D array for x axis
              param_dict, *,
              vertical_offset=0,
              legend_param=None, # Parameter name for legend naming
              legend_fmt="{plotkind} {value}", # Format string for legend
              legend_kwargs={},
              plot_kwargs={},
              ydata=None, # 1 or 2D-array for y axis. The x axis should be the last axis
              ):

        ## Initialize adapters for this run
        #
        # Every adapter goes through param_dict and sets `has_adapter`
        # for parameters it wants to control.
        #

        param_adapters = []
        fit_vars = []
        fit_bounds = ([], [])

        for adapter_class in self.adapters:
            param_adapters.append(adapter_class(param_dict, self.bounds, fit_vars, fit_bounds))

        param_adapters.append(DefaultAdapter(param_dict, self.bounds, fit_vars, fit_bounds))

        ## wh_args is a list of arguments to workhorse(*wh_args)
        # It contains constant parameters and initial values
        # for fitted parameters
        # In the multifit case, wh_args is a tuple of lists.

        wh_args = [None for p in param_dict.values() if p.kind == ParameterKind.POSITIONAL]
        wh_kwargs = {}

        def fill_wh_args(args, kwargs, pdict):
            for p in pdict.values():
                if p.kind == ParameterKind.POSITIONAL:
                    args[p.index] = p.current_value
                elif p.kind == ParameterKind.KEYWORD:
                    kwargs[p.name] = p.current_value

        fill_wh_args(wh_args, wh_kwargs, param_dict)

        multi_fit_N = max(p.multiplicity for p in param_dict.values())
        multi_fit = multi_fit_N > 1

        exp_labels = []
        fit_labels = []

        for adapter in param_adapters:
            adapter.read_guesses(fit_vars)
        if multi_fit:
            wh_data = zeros((multi_fit_N, len(xdata)))
            for i in range(multi_fit_N):
                for adapter in param_adapters:
                    adapter.read_single_guess(fit_vars, i)
                fill_wh_args(wh_args, wh_kwargs, param_dict)

                wh_data[i] = self.workhorse(xdata, *wh_args, **wh_kwargs)

                legend_value = param_dict[legend_param].current_value \
                                 if legend_param is not None else (i+1)

                exp_labels.append(
                    legend_fmt.format(plotkind="Experiment", value=legend_value))
                fit_labels.append(
                    legend_fmt.format(plotkind="Fit", value=legend_value))
        else:
            wh_data = zeros_like(xdata)
            for adapter in param_adapters:
                adapter.read_single_guess(fit_vars, 0)
            fill_wh_args(wh_args, wh_kwargs, param_dict)

            wh_data = self.workhorse(xdata, *wh_args, **wh_kwargs)

            legend_value = param_dict[legend_param].current_value if legend_param is not None else 1

            exp_labels.append(
                legend_fmt.format(plotkind="Experiment", value=legend_value))
            fit_labels.append(
                legend_fmt.format(plotkind="Fit", value=legend_value))

        if 'linestyle' in plot_kwargs:
            linestyle = plot_kwargs['linestyle']
            del plot_kwargs['linestyle']
        else:
            linestyle = '-'

        if ydata is not None:
            if 'marker' in plot_kwargs:
                marker = plot_kwargs['marker']
                del plot_kwargs['marker']
            else:
                marker = '.'
            colors = self.plot_waterfall(xdata, ydata,
                                         vertical_offset=vertical_offset,
                                         linestyle='',
                                         marker=marker,
                                         labels=exp_labels,
                                         **plot_kwargs)
        else:
            colors = None

        self.plot_waterfall(xdata, wh_data,
                            colors=colors,
                            vertical_offset=vertical_offset,
                            linestyle=linestyle,
                            labels=fit_labels,
                            **plot_kwargs)

        plt.legend(**legend_kwargs)


    def plot(self,
             xdata, # 1D array for x axis
             ydata, # 1 or 2D-array for y axis. The x axis should be the last axis
             *args, # Any positional parameters to pass to the calculating function
             coeff_dict={}, # Parameters from an earlier run
             const_array_params=[],
             vertical_offset=0,
             legend_param=None, # Parameter name for legend naming
             legend_fmt="{plotkind} {value}", # Format string for legend
             legend_kwargs={},
             plot_ydata=True,
             plot_kwargs={},
             **kwargs # Parameters to fit or not fit
             ):
        """Plot the workhorse with given parameters.
        This function takes an arbitrary number of arguments according to the following:

        `xdata` is a 1D array of x-axis values (shape `(m,)`).
        `ydata` is a 1D or 2D array of data values (shape `(n, m)`).

        `plot_ydata` will plot a scatterplot for the supplied data if True.
        `plot_vertical_offset` is a number to use for offsetting different curves in the final plot.
        `plot_legend_param` is a name of a workhorse argument.
        `plot_legend_fmt` is a label to attach to every plot. It is a formatting string
          with two possible arguments: *plotkind* that is a string 'Initial', 'Experiment' or 'Fit',
          and *value* that is the value of the fit parameter selected by `plot_legend_param`.
        `legend_kwargs` is a dictionary that will be passed to `pylab.legend()` as keyword arguments.
          One can use it to specify e.g. legend location on the graph.
        `plot_kwargs` is a dictionary that will be passed to `pylab.plot()` as keyword arguments.
          One can use it to specify e.g. a linestyle.       

        """

        ## This part decides wherether X and Y are in good shape

        y_shape = ydata.shape
        if len(xdata) != y_shape[-1]:
            raise ValueError("X shape {xdata.shape} and Y shape {ydata.shape} do not match.")
        if len(y_shape) > 1:
            multi_fit_N = prod(y_shape[:-1])
        else:
            multi_fit_N = 1

        # First we put together all the parameters we now know about
        param_dict = self.fill_param_dict(args, kwargs,
                                          [], const_array_params,
                                          coeff_dict, multi_fit_N)

        return self._plot(
            xdata, param_dict,
            ydata=ydata if plot_ydata else None, vertical_offset=vertical_offset,
            legend_param=legend_param, # Parameter name for legend naming
            legend_fmt=legend_fmt, # Format string for legend
            legend_kwargs=legend_kwargs,
            plot_kwargs=plot_kwargs)
