import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class FFrame(object):
    """FFrame is (for now) a 1D piecewise discretizer for arbitrary domain and image granularity.
    Args:
        domain_min (float): Minimum of the domain.
        domain_max (float): Maximum of the domain.
        func (:obj:callable): Function to discretize.
        outside the domain. Defaults to True.
        domain_gran (float): Domain granularity. Defaults to 5.
        image_gran (float): Image granularity. Defaults to 0.5.
    Attributes:
        domain (:obj:array of :obj:float): Discretized domain.
        func_domain (:obj:array of :obj:float): Domain used for interpolator.        
        values (:obj:array of :obj:float): Values of func interpolated on domain.
        discrete_func (:obj:array of :obj:float): Discretized function.
    """

    interp_values = []
    func_domain = []


    
    def __init__(self, domain_min, domain_max, func,
                domain_gran=5, image_gran=0.5,
                ):

        self.domain_gran = domain_gran
        self.image_gran = image_gran
        # Create the domain and image values
        self.func = func
        self.domain_min = domain_min
        self.domain_max = domain_max

        domain = np.arange(start=domain_min, stop=domain_max+self.domain_gran, step=self.domain_gran)
        self.domain = domain

        
        # Define the allowed values given the chosen granularities

    def functional_method(self):
        for x in self.domain:
            yield np.round(self.func(x) / self.image_gran) * self.image_gran

    def lut_method(self):

        values = self.func(self.domain)
        max_val = np.max(values)
        # if extrapolate: max_val += self.image_gran
        min_val = np.round(np.min(values) / self.image_gran) * self.image_gran
        allowed_values = np.arange(start=min_val, stop=max_val+self.image_gran, step=self.image_gran)
        return self.chunk(allowed_values, values)
        
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

    def compare_functions(self, x_values, ax=None, show=True):
        """Plot the discretized and original versions of the function.
        Returns:
            ax (:obj:axis): Axis object.
        """
        
        # discrete_func = self.lut_method()
        interp_image = self.func(x_values)
        functional_vals = list(self.functional_method())

        if ax:
            # ax.plot(self.domain, discrete_func, label='lut', color='b')
            ax.plot(self.domain, functional_vals, label='functional', color='k')
            ax.plot(x_values, interp_image, label='Input values', color='r')
        else:
            # plt.plot(self.domain, discrete_func, label='lut', color='b')
            plt.plot(self.domain, functional_vals, label='functional', color='k')
            plt.plot(x_values, interp_image, label='Input values', color='r')

        ax = plt.gca()
            
        if show:
            plt.legend()
            plt.show()
        
        return ax


if __name__ == "__main__":
    from timerit import Timerit

    f = FFrame(5, 120, lambda x: 100*np.sin(x)+np.cos(x), domain_gran=5, image_gran=2)

    f.compare_functions(np.linspace(0,120,num=1000))

    if True:
        print("Functional method:")
        for _ in Timerit(num=100, verbose=2):
            f.functional_method()

        print("\nLUT method:")
        for _ in Timerit(num=100, verbose=2):
            f.lut_method()

    # t = Timer(f.functional_method, number=1000)
    # print(t)
    # f.lut_method()
