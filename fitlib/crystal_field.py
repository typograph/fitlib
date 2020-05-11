'Adapters for working with the JLS library'

from .adapters import BuildingAdapter

class ZTAdapter_base(BuildingAdapter):
    """This will fit magnetic field in either cartesian or spherical components"""

    def __new__(cls, *args):
        if cls == ZTAdapter_base:
            raise ValueError("ZTAdapter_base cannot be initialized directly")
        else:
            return super().__new__(cls)

    def __init__(self, *args):
        super().__init__(*args)

        self.atom = self.get_parameter(self.quantum_system_name).initial_value
        self.ZT = self.__class__.zt_accessor(self.atom)

        self.cartesian = 'Bx' in self.param_dict or 'By' in self.param_dict or 'Bz' in self.param_dict
        self.spherical = 'Br' in self.param_dict or 'Btheta' in self.param_dict or 'Bphi' in self.param_dict
        
        if self.cartesian and self.spherical:
            raise ValueError('Cannot fit in cartesian and spherical coordinates simultaneously')
            
        if self.cartesian:
            for field in ['Bx', 'By', 'Bz']:
#                if field in self.param_dict:
                    self.add_parameter(field, self.ZT.__getattribute__(field))
                    
        elif self.spherical:
            for field in ['Br', 'Btheta', 'Bphi']:
#                if field in self.param_dict:
                    self.add_parameter(field, self.ZT.__getattribute__(field))

        self.add_parameter('g', self.ZT.getg())

    def build_values(self, i):
        if self.parameters['g'].fitted: # Just in case nothing is fitted
            self.ZT.setg(self.parameters['g'].value_at(i))
            
        if self.cartesian:
            self.ZT.setBxyz(self.parameters['Bx'].value_at(i),
                            self.parameters['By'].value_at(i),
                            self.parameters['Bz'].value_at(i))
        
        elif self.spherical:
            self.ZT.setBrtp(self.parameters['Br'].value_at(i),
                            self.parameters['Btheta'].value_at(i),
                            self.parameters['Bphi'].value_at(i))
        
def ZTAdapter(quantum_system_name, zt_accessor=lambda o: o.ZT):
    return type("ZTAdapter_{}".format(quantum_system_name),
                (ZTAdapter_base,),
                {"quantum_system_name":quantum_system_name,
                 "zt_accessor":zt_accessor})

class CFAdapter_base(BuildingAdapter):

    def __new__(cls, *args):
        if cls == CFAdapter_base:
            raise ValueError("CFAdapter_base cannot be initialized directly")
        else:
            return super().__new__(cls)

    def __init__(self, *args):
        super().__init__(*args)

        self.atom = self.get_parameter(self.quantum_system_name).initial_value
        self.CF = self.__class__.cf_accessor(self.atom)
        self.fitted_orders = []

        for j, (n, m) in enumerate(self.CF.orders):
            pname = "CF_{}_{}".format(n,m)
            if pname in self.param_dict:
                self.add_parameter(pname, self.CF.coeff[j])
                self.fitted_orders.append((pname, j))

    def build_values(self, guess, i=None):
        for pname, j in self.fitted_orders:
            self.CF.coeffs[j] = self.param_dict[pname].current_value
        self.CF.makeNotReady()

def CFAdapter(quantum_system_name, cf_accessor=lambda o: o.CF):
    return type("CFAdapter_{}".format(quantum_system_name),
                (CFAdapter_base,),
                {"quantum_system_name":quantum_system_name,
                 "cf_accessor":cf_accessor})
