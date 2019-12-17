import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class FFrame(object):
    """FFrame is (for now) a 1D piecewise discretizer for arbitrary domain and image granularity.
    Args:
        domain_min (float): Minimum of the domain.
        domain_max (float): Maximum of the domain.
        func (:obj:callable): Function to discretize.
        funcsteps (int, optional): Number of steps to interpolate function on. Defaults to 100.
        extrapolate (bool, optional): Whether or not to include the points immediately 
        outside the domain. Defaults to True.
        domain_gran (float): Domain granularity. Defaults to 5.
        image_gran (float): Image granularity. Defaults to 0.5.
    Attributes:
        domain (:obj:array of :obj:float): Discretized domain.
        interpolator (:obj:callable): 1dinterp object.
        func_domain (:obj:array of :obj:float): Domain used for interpolator.        
        values (:obj:array of :obj:float): Values of func interpolated on domain.
        discrete_func (:obj:array of :obj:float): Discretized function.
    """

    interp_values = []
    func_domain = []


    
    def __init__(self, domain_min, domain_max, func,
                funcsteps=100, extrapolate=True,
                domain_gran=5, image_gran=0.5,
                func_values, ):

        self.domain_gran = domain_gran
        self.image_gran = image_gran
        # Create the domain, interpolator, and interpolated values
        self.func = func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.funcsteps = funcsteps

        domain = np.arange(start=domain_min, stop=domain_max+self.domain_gran, step=self.domain_gran)
        self.domain = domain

        self.func_domain = np.linspace(self.domain_min, self.domain_max, num=self.funcsteps)
        self.values = self.func(self.func_domain)
        
        # Define the allowed values given the chosen granularities
        max_val = np.max(values)
        if extrapolate: max_val += self.image_gran
        min_val = self.get_min_val(np.min(values), extrapolate=extrapolate)
        allowed_values = np.arange(start=min_val, stop=max_val+self.image_gran, step=self.image_gran)
        self.discrete_func = self.chunk(allowed_values, values)

    def chunk(self, allowed_values, values, steps=10):
        """ Round values to nearest allowed_value. """

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            from itertools import tee

            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        last_value_allowed = values[-1] in allowed_values
        allowed_domain = np.linspace(allowed_values[0], allowed_values[-1], steps)
        allowed_pairs = list(pairwise(allowed_domain))
        allowed_partition = [[value for value in allowed_values 
                                if (value>=mini and value<maxi)]
                            for mini, maxi in allowed_pairs]
        if last_value_allowed: allowed_partition[-1].append(values[-1])
        allowed_values = None
        chunked = []
        for value in values:
            for minimaxi, partition in zip(allowed_pairs, allowed_partition):
                mini, maxi = minimaxi
                if value >= mini and value < maxi:
                    dists = [(abs(value-allowed), allowed) for allowed in partition]
                    distsort = sorted(dists, key=lambda pair:pair[0])
                    chunked += [distsort[0][1]]

        if last_value_allowed: chunked.append(values[-1])
        return chunked
    
    def get_min_val(self, minval, extrapolate=True):
        """ Returns multiple of image_gran nearest to minval """        
        from math import ceil, floor

        if minval == 0:
            return 0
            
        elif minval < 0:
            if extrapolate:
                return floor(minval / self.image_gran) * self.image_gran
            else: 
                return ceil(minval / self.image_gran) * self.image_gran

        elif minval > 0:
            if extrapolate:
                return ceil(minval / self.image_gran) * self.image_gran
            else: 
                return floor(minval / self.image_gran) * self.image_gran

    def agglomerated(self):
        """Agglomerate neighboring image values if they are the same 
        Returns:
            agglomerated_steps (:obj:array of :obj:float): Domain values for agglomerated function values.
            agglomerated_values (:obj:array of :obj:float): Agglomerated image values.
        """

        agglomerated_values = [self.discrete_func[0]]
        agglomerated_steps = [self.domain[0]]

        for step, value in zip(self.domain[1:], self.discrete_func[1:]):
            if value == agglomerated_values[-1]:
                agglomerated_values.append(value)
                agglomerated_steps.append(step)
        
        return np.array(agglomerated_steps), np.array(agglomerated_values)

    def compare_functions(self, ax=None, show=True):
        """Plot the discretized and interpolated versions of the function.
        Returns:
            ax (:obj:axis): Axis object.
        """
        
        interp_image = self.interpolator(self.func_domain)

        if ax:
            ax.plot(self.domain, self.discrete_func, label='discretized', color='b')
            ax.plot(self.func_domain, interp_image, label='interpolated', color='r')
        else:
            plt.plot(self.domain, self.discrete_func, label='discretized', color='b')
            plt.plot(self.func_domain, interp_image, label='interpolated', color='r')

        ax = plt.gca()
            
        if show:
            plt.legend()
            plt.show()
        
        return ax

if __name__ == "__main__":

    f = FFrame(0, 120, lambda x:10*np.sin(x),
               funcsteps=1000, domain_gran=2)
    f.compare_functions(show=True)
