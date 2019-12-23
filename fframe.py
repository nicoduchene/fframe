import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from timerit import Timerit

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

        self.domain = np.arange(start=domain_min, stop=domain_max+self.domain_gran, step=self.domain_gran)


    def functional_method(self):
        """ Functional implementation. """
        for x in self.domain:
            yield np.round(self.func(x) / self.image_gran) * self.image_gran
    
    def init_lut(self, image_chunks=10):
        """ Initialize data structures. """
        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            from itertools import tee

            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        values = self.func(self.domain)        
        max_val = np.max(values)
        # if extrapolate: max_val += self.image_gran
        min_val = np.round(np.min(values) / self.image_gran) * self.image_gran
        allowed_values = np.arange(start=min_val, stop=max_val+self.image_gran, step=self.image_gran)

        last_value_allowed = values[-1] in allowed_values

        allowed_domain = np.linspace(allowed_values[0], allowed_values[-1], image_chunks)
        allowed_pairs = list(pairwise(allowed_domain))
        allowed_partition = [[value for value in allowed_values 
                                if (value>=mini and value<maxi)]
                            for mini, maxi in allowed_pairs]
        if last_value_allowed: allowed_partition[-1].append(values[-1])

        self.values = values
        self.allowed_pairs = allowed_pairs
        self.allowed_partition = allowed_partition
        self.last_value_allowed = last_value_allowed

    def lut_method(self):
        """ Round values to nearest allowed_value. """
        
        chunked = []
        for value in self.values:
            for minimaxi, partition in zip(self.allowed_pairs, self.allowed_partition):
                mini, maxi = minimaxi
                if value >= mini and value < maxi:
                    dists = [(abs(value-allowed), allowed) for allowed in partition]
                    distsort = sorted(dists, key=lambda pair:pair[0])
                    chunked += [distsort[0][1]]

        if self.last_value_allowed: chunked.append(values[-1])
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

    def plot_functions(self, x_values, ax=None, show=True):
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

    def time_functional(self, numloops=1000):
        """ Time the functional method """
        t = Timerit(num=numloops)
        for _ in t:
            self.functional_method()
        results = {'method': 'functional',
                   'mean': [t.mean()],
                   'std': [t.std()]
                   }
        return results 
        
    def time_lut(self, numloops=1000):
        """ Time LUT method"""
        t = Timerit(num=numloops)
        for _ in t:
            self.lut_method()
        results = {'method': 'lut',
                   'mean': [t.mean()],
                   'std': [t.std()]
                   }
        return results

import pandas as pd

class Analysis(object):
    """Group together the analysis results and present them."""

    def __init__(self, ax):
        self.df = pd.DataFrame()
        self.ax = ax

    def add_results(self, performance_dict):
        self.df = self.df.append(pd.DataFrame(performance_dict))

    def plot_results(self, x_label='func_label', y_labels=['mean', 'std']):
        self.df.plot(x_label, y_labels, kind='bar', subplots=False, logy=True, ax=self.ax)


def analyze(a_func, a_lut, func, d_min, d_max, d_gran, i_gran, label):
    """Analyze performance of functional and lut methods for given args.
    Updates the two analysis objects that are input.
    Args:
        a_func (:obj:Analysis): Analysis object for functional implementation.
        a_lut (:obj:Analysis): Analysis object for lut implementation.
        func (:obj:callable): Function to discretize.
        d_min (float): Domain min.
        d_max (float): Domain max.
        d_gran (float): Domain granularity.
        i_gran (float): Image granularity.
        label (str): Label root to name func_labels in analysis dataframes.
    """

    f = FFrame(d_min, d_max, func, domain_gran=d_gran, image_gran=i_gran)
    
    results = f.time_functional()
    results['func_label'] = label
    a_func.add_results(results)    

    f.init_lut(image_chunks=10)
    results = f.time_lut()
    results['func_label'] = label+'_c10'
    a_lut.add_results(results)

    f.init_lut(image_chunks=25)
    results = f.time_lut()
    results['func_label'] = label+'_c25'
    a_lut.add_results(results)

    f.init_lut(image_chunks=50)
    results = f.time_lut()
    results['func_label'] = label+'_c50'
    a_lut.add_results(results)

    del(f)

def analyze_x3sinx():

    _, ax = plt.subplots(1,2)
    a_func = Analysis(ax=ax[0])
    a_lut = Analysis(ax=ax[1])

    func = lambda x: np.sin(x)*x**3
    """ x**3sin(x) from 0 to 10 """
    analyze(a_func, a_lut, func, 0, 10, 0.1, 20, 'd10')
    
    """ x**3sin(x) from 0 to 100 """
    analyze(a_func, a_lut, func, 0, 100, 1, 20000, 'd100')
    
    """ x**3sin(x) from 0 to 1000 """
    analyze(a_func, a_lut, func, 0, 1000, 10, 2000000, 'd1000')
    
    a_func.plot_results()
    a_lut.plot_results()
    plt.show()

def analyze_sinx_over_x():
    
    _, ax = plt.subplots(1,2)
    a_func = Analysis(ax=ax[0])
    a_lut = Analysis(ax=ax[1])

    func = lambda x: np.sin(x)/x
    """ x**3sin(x) from 0.1 to 10 """
    analyze(a_func, a_lut, func, 0.1, 10, 0.1, 0.01, 'd10')
    
    """ x**3sin(x) from 0 to 100 """
    analyze(a_func, a_lut, func, 0.1, 100, 1, 0.01, 'd100')
    
    """ x**3sin(x) from 0 to 1000 """
    analyze(a_func, a_lut, func, 0.1, 1000, 10, 0.01, 'd1000')
    
    a_func.plot_results()
    a_lut.plot_results()
    plt.show()

if __name__ == "__main__":

    # analyze_x3sinx()
    analyze_sinx_over_x()
    f = FFrame(0.1, 100, lambda x: np.sin(x)/x, domain_gran=1, image_gran=0.01)
    f.plot_functions(np.linspace(0.1,100,1000))
    plt.show()
