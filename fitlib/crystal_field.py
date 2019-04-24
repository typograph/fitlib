'Adapters for working with the JLS library'

from .adapters import EasyAdapter

class ZTAdapter(EasyAdapter):
    """"""

    def __new__(cls, *args):
        if cls == ZTAdapter:
            raise ValueError("ZTAdapter cannot be initialized directly")
        else:
            return super().__new__(cls, *args)

    def __init__(self, *args):
        super().__init__(*args)

        if self.quantum_system_name not in self.params_dict:
            raise ValueError("{} argument is needed for crystal field"
                             " but it is not in the workhorse signature".format(
                                 self.quantum_system_name))

        if not self.params_dict[self.quantum_system_name].has_initial_value:
            raise ValueError("{} argument is needed for crystal field to be fitted"
                             " but non value was supplied".format(self.quantum_system_name))

        self.atom = self.params_dict[self.quantum_system_name].initial_value
        self.ZT = self.zt_accessor(self.atom)

        self.fit_fields = []

        for field in ['Bx', 'By', 'Bz', 'Br', 'Btheta', 'Bphi']:
            if field in self.params_dict:
                self.add_parameter(field, self.ZT.__getattribute__(field))

        self.add_parameter('g', self.ZT.gFactor)

    def read_guess(self, guess, i=None):
        super().fill_params(i)
        ## This needs some thinking
#        self.ZT.setg(self.parameters['g'])
        self.ZT.makeNotReady()

def gen_ZTAdapter(quantum_system_name, zt_accessor=lambda o: o.ZT):
    return type("ZTAdapter_{}".format(quantum_system_name),
                (ZTAdapter,),
                {"quantum_system_name":quantum_system_name,
                 "zt_accessor":zt_accessor})

class CFAdapter(EasyAdapter):

    def __new__(cls, *args):
        if cls == CFAdapter:
            raise ValueError("CFAdapter cannot be initialized directly")
        else:
            return super().__new__(cls, *args)

    def __init__(self,
                 multi_fit_N, # 2D fit or not?
                 params_dict,
                 params_bounds,
                 ):
        super().__init__(multi_fit_N,
                         params_dict,
                         params_bounds
                         )

        if self.quantum_system_name not in params_dict:
            raise ValueError("{} argument is needed for crystal field"
                             " but it is not in the workhorse signature".format(
                                 self.quantum_system_name))

        if not params_dict[self.quantum_system_name].has_initial_value:
            raise ValueError("{} argument is needed for crystal field to be fitted"
                             " but non value was supplied".format(self.quantum_system_name))

        self.atom = params_dict[self.quantum_system_name].initial_value
        self.CF = self.cf_accessor(self.atom)
        self.fitted_orders = []

        for j, (n, m) in enumerate(self.CF.orders):
            pname = "CF_{}_{}".format(n,m)
            if pname in params_dict:
                self.add_parameter(pname, self.CF.coeff[j])
                self.fitted_orders.append((pname, j))

    def read_guess(self, guess, i=None):
        super().read_guess(guess, i)
        if i is not None:
            for pname, j in self.fitted_orders:
                self.CF.coeffs[j] = self.params_dict[pname].current_value
            self.CF.makeNotReady()

def gen_CFAdapter(quantum_system_name, cf_accessor=lambda o: o.CF):
    return type("CFAdapter_{}".format(quantum_system_name),
                (CFAdapter,),
                {"quantum_system_name":quantum_system_name,
                 "cf_accessor":cf_accessor})
