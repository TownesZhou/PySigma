"""
    Utility functions
"""
import torch
import torch.distributions
from torch.distributions import Distribution, ExponentialFamily
from torch.distributions.constraints import Constraint, integer_interval
from collections.abc import Iterable
import math


def intern_name(name: str, struc_type: str):
    """
        Add prefix and brackets to transform user provided structure name to internal name
    :param name:    Structure name
    :param type:    one of "type", "predicate", or "conditional
    :return:        processed name
    """
    assert struc_type in ["type", "predicate", "conditional"], "unknown type for processing structure name"
    assert isinstance(name, str)
    if struc_type is "type":
        return "TYPE_[" + name.upper() + "]"
    elif struc_type is "predicate":
        return "PRED_[" + name.upper() + "]"
    elif struc_type is "conditional":
        return "COND_[" + name.upper() + "]"


def extern_name(name: str, struc_type: str):
    """
        Inverse operation of intern_name
    """
    assert struc_type in ["type", "predicate", "conditional"], "unknown type for processing structure name"
    assert isinstance(name, str)
    if struc_type is "type":
        assert name.find("TYPE_") >= 0
    if struc_type is "predicate":
        assert name.find("PRED_[") >= 0
    if struc_type is "conditional":
        assert name.find("COND_[") >= 0
    return name[6:-1]


# TODO: Global dictionary that designates which PyTorch's distribution class is finite discrete
FINITE_DISCRETE_CLASSES = [
    torch.distributions.Categorical,
    torch.distributions.Bernoulli,
    torch.distributions.Binomial,
]


# TODO: DistributionServer class
class DistributionServer:
    """
        Serving distribution class dependent utilities
            - Conversion between PyTorch distribution parameters and distribution instance:
                    param2dist(), dist2param()
            - Translation between PyTorch distribution parameters and natural parameters for exponential family
                distribution:
                    natural2exp_dist(), exp_dist2natural()
            - Get vector of moments from a given distribution instance:
                    get_moments()
            - Draw particles from  distribution instance:
                    draw_particles()
            - Get log probability density from given particles:
                    log_pdf()

        Certain distribution classes require special handling, for example for those categorized as finite discrete,
            particle values will be drawn uniformly, covering every value in the RV's value domain once and only once,
            while assigning each particle its probability mass as its particle weight.
        Therefore we delegate all such special handlings to this class on an individual basis.

        Note that input and output will conform to the format understandable by PyTorch's distribution class. To
            translate to and from formats compatible to PySigma's predicate knowledge, use KnowledgeTranslator class
    """
    @classmethod
    def param2dist(cls, dist_class, params, b_shape, e_shape):
        """
            Conversion from PyTorch distribution parameters to distribution instance
            Return a distribution instance
        """
        assert isinstance(params, torch.Tensor)
        assert issubclass(dist_class, torch.distributions.Distribution)
        assert isinstance(b_shape, torch.Size)
        assert isinstance(e_shape, torch.Size)

        if dist_class not in cls.dict_param2dist.keys():
            raise NotImplementedError("Conversion from parameters to distribution instance for distribution class '{}' "
                                      "not yet implemented".format(dist_class))

        dist = cls.dict_param2dist[dist_class](params)
        if dist.batch_shape != b_shape or dist.event_shape != e_shape:
            raise ValueError("The shape of the generated distribution {} does not match the provided shape. "
                             "Provided (batch_shape, event_shape) == ({}, {}), but instead got ({}, {})."
                             .format(str(dist), b_shape, e_shape, dist.batch_shape, dist.event_shape))
        return dist

    @classmethod
    def dist2param(cls, dist):
        """
            Conversion from distribution instance to PyTorch distribution parameters
            Return a parameter tensor
        """
        assert isinstance(dist, Distribution)
        dist_class = type(dist)

        if dist_class not in cls.dict_dist2param.keys():
            raise NotImplementedError("Conversion from distribution instance to parameters for distribution class '{}' "
                                      "not yet implemented".format(dist_class))
        return cls.dict_dist2param[dist_class](dist)

    @classmethod
    def natural2exp_param(cls, dist_class, natural_params, b_shape, param_shape):
        """
            Translation from natural parameters to PyTorch distribution parameters for exponential family
            Return a parameter tensor
        """
        assert isinstance(natural_params, torch.Tensor)
        assert issubclass(dist_class, ExponentialFamily)
        assert isinstance(b_shape, torch.Size)
        assert isinstance(param_shape, torch.Size)

        if dist_class not in cls.dict_natural2exp_param.keys():
            raise NotImplementedError("Translation from natural parameters to PyTorch distribution parameters for "
                                      "distribution class '{}' not yet implemented".format(dist_class))
        return cls.dict_natural2exp_param[dist_class](natural_params, b_shape, param_shape)

    @classmethod
    def exp_param2natural(cls, dist_class, exp_params, b_shape, param_shape):
        """
            Translation from PyTorch distribution parameters to natural parameters for exponential family
            Return a parameter tensor
        """
        assert isinstance(exp_params, torch.Tensor)
        assert issubclass(dist_class, ExponentialFamily)
        assert isinstance(b_shape, torch.Size)
        assert isinstance(param_shape, torch.Size)

        if dist_class not in cls.dict_exp_param2natural.keys():
            raise NotImplementedError("Translation from natural parameters to PyTorch distribution parameters for "
                                      "distribution class '{}' not yet implemented".format(dist_class))
        return cls.dict_exp_param2natural[dist_class](exp_params, b_shape, param_shape)

    @classmethod
    def natural2exp_dist(cls, dist_class, natural_params, b_shape, e_shape, param_shape):
        """
            Composition of param2dist() with natural2exp_param()
            Return a distribution instance
        """
        param = cls.natural2exp_param(dist_class, natural_params, b_shape, param_shape)
        dist = cls.param2dist(dist_class, param, b_shape, e_shape)
        return dist

    @classmethod
    def exp_dist2natural(cls, dist):
        """
            Composition of exp_param2natural() with dist2param()
            Return a parameter tensor
        """
        exp_param = cls.dist2param(dist)
        natural = cls.exp_param2natural(type(dist), exp_param, dist.batch_shape,
                                        exp_param.shape[len(dist.batch_shape):])
        return natural

    @classmethod
    def get_moments(cls, dist, n_moments):
        """
            Get vector of moments from a given distribution instance
        """
        assert isinstance(dist, Distribution)
        assert isinstance(n_moments, int) and n_moments > 0

        dist_class = type(dist)
        if dist_class not in cls.dict_get_moments.keys():
            raise NotImplementedError("Get moments method for distribution class '{}' not yet implemented"
                                      .format(dist_class))
        return cls.dict_get_moments[dist_class](dist, n_moments)

    @classmethod
    def draw_particles(cls, dist, num_particles):
        """
            Draw a given number of particles from the given distribution instance. Return a tuple:
                    (particles, weights, sampling_log_densities)

            Special handling for certain distribution classes on an individual basis, for example for finite discrete
                distributions.

            Particles drawn are in the format compatible with PyTorch's distribution class
        """
        assert isinstance(dist, Distribution)
        assert isinstance(num_particles, int)

        dist_class = type(dist)
        if dist_class not in cls.dict_draw_particles.keys():
            raise NotImplementedError("Draw particles method for distribution class '{}' not yet implemented"
                                      .format(dist_class))
        return cls.dict_draw_particles[dist_class](dist, num_particles)

    @classmethod
    def log_pdf(cls, dist, particles):
        assert isinstance(dist, Distribution)
        assert isinstance(particles, torch.Tensor)

        dist_class = type(dist)
        if dist_class not in cls.dict_log_pdf.keys():
            raise NotImplementedError("Get log pdf method for distribution class '{}' not yet implemented"
                                      .format(dist_class))
        return cls.dict_log_pdf[dist_class](dist, particles)

    """
        DEFAULT methods that may be applicable to multiple general distribution classes
    """
    @classmethod
    def _default_get_moments(cls, dist, n_moments):
        """
            Default method for getting moments, but only supports up to second order moment (i.e. X^2)
        """
        assert n_moments <= 2

        mean = dist.mean
        if n_moments == 1:
            return mean
        else:
            square = dist.variance + mean ** 2
            # Stack mean and square to insert a new last dimension
            result = torch.stack([mean, square], dim=len(mean.shape))
            return result

    @classmethod
    def _default_draw_particles(cls, dist, num_particles):
        s_shape = torch.Size([num_particles])
        particles = dist.sample(sample_shape=s_shape)
        weights = 1  # uniform weights
        sampling_log_densities = dist.log_prob(value=particles)

        return particles, weights, sampling_log_densities

    @classmethod
    def _default_log_pdf(cls, dist, particles):
        log_pdf = dist.log_prob(value=particles)
        return log_pdf

    """
        Categorical distribution
    """
    @classmethod
    def _categorical_param2dist(cls, params):
        """
            For categorical, params assumed to be fed as the 'probs' attribute
        """
        dist = torch.distributions.Categorical(probs=params)
        return dist

    @classmethod
    def _categorical_dist2param(cls, dist):
        return dist.probs

    @classmethod
    def _categorical_draw(cls, dist, num_particles):
        """
            Draw categorical particles. Span of RV domain is inferred from last dimension of the distribution instance's
                'probs' attribute. Particles will be drawn uniformly covering every value in the RV's domain once and
                only once, while their probability mass will be assigned as the particle weights respectively.
        """
        assert isinstance(dist, torch.distributions.Categorical)
        span = dist.probs.shape[-1]
        b_shape = dist.batch_shape
        s_shape = torch.Size(num_particles)

        particles = torch.ones(s_shape + b_shape)
        for i in range(span):
            particles[i] = particles[i] * i

        # Weights obtained from probs attribute, by simply permuting the last dimension to first dimension
        n_dims = len(dist.probs.shape)
        dims = [n_dims-1, ] + [i for i in range(n_dims - 1)]
        weights = dist.probs.clone().permute(dims)      # clone to prevent accidental in-place value change

        # sampling log density obtained from API
        sampling_log_densities = dist.log_prob(value=particles)

        return particles, weights, sampling_log_densities

    """
        distribution class dependent method pointer
    """
    dict_draw_particles = {
        torch.distributions.Categorical: _categorical_draw
    }
    dict_log_pdf = {

    }
    dict_param2dist = {

    }
    dict_dist2param = {

    }
    dict_natural2exp_param = {

    }
    dict_exp_param2natural = {

    }
    dict_natural2exp_dist = {

    }
    dict_exp_dist2natural = {

    }
    dict_get_moments = {

    }



# TODO: Particle knowledge translator class
class KnowledgeTranslator:
    """
        knowledge translator class. Translate knowledge tensors between the forms understandable by Predicate and that
            understandable by PyTorch's distribution class. This includes both event particle tensors and parameter
            tensors.

        Different distribution class requires different handling, but in general values for multiple random variables
            (potentially of different sizes) will be concatenated together to form the last dimension of event samples
            that can be interpreted by PyTorch. Therefore when translating back to Predicate knowledge format, what
            is returned is a TUPLE of tensors, size of which equal to the number of RVs, and each tensor within which
            corresponds to one RV's value assignment respectively. Similarly, what is taken when translating from
            Predicate knowledge to PyTorch's knowledge is also a tuple of tensors

        A translator instance should be instantiated and hold by each Predicate. When instantiated, also check if
            provided var_sizes and var_constraints are compatible with dist_class. This is also distribution class
            dependent therefore needs individual implementation.
    """

    def __init__(self, dist_class, var_sizes, var_constraints):
        """
            Instantiate a translator

            :param dist_class: A subclass of torch.distributions.Distributions
            :param var_sizes:  A sequence of integers, denoting the sizes of a predicate's random variables. The order
                               will be respected when concatenating event values.
            :param var_constraints: A sequence of torch.distributions.constraints.Constraint object. Each denoting the
                                    value constraint of the corresponding random variable.
        """
        # distribution-dependent translation method pointer. Indexed by distribution class
        self.dict_2format_check = {
            torch.distributions.Categorical: self._categorical_format_check
        }
        self.dict_2torch_event = {
            torch.distributions.Categorical: self._categorical_2torch_event
        }
        self.dict_2pred_event = {
            torch.distributions.Categorical: self._categorical_2pred_event
        }
        self.dict_2torch_param = {
            torch.distributions.Categorical: self._categorical_2torch_param
        }
        self.dict_2pred_param = {
            torch.distributions.Categorical: self._categorical_2pred_param
        }

        assert issubclass(dist_class, Distribution)
        assert isinstance(var_sizes, Iterable) and all(isinstance(size, int) for size in var_sizes)
        assert isinstance(var_constraints, Iterable) and all(isinstance(c, Constraint) for c in var_constraints)
        assert len(var_sizes) == len(var_constraints)

        self.dist_class = dist_class
        self.var_sizes = var_sizes
        self.var_constraints = var_constraints

        self.num_vars = len(var_sizes)

    def _format_check(self):
        if self.dist_class not in self.dict_2format_check.keys():
            raise NotImplementedError("Format check for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        self.dict_2format_check[self.dist_class]()

    def event2torch_event(self, particles):
        """
            Translate event particles from the format understandable by Predicate to the format understandable by
                PyTorch
            Take as input a tuple of torch tensors. Check that sizes of last dimensions equal to sizes of RVs
            Return a single torch tensor.
        """
        # Do not check particle last dimension's size if var size is 1, because likely a dimension of size 1 will be
        #   squeezed
        assert isinstance(particles, tuple) and len(particles) == self.num_vars and \
               all(isinstance(p, torch.Tensor) for p in particles) and \
               all(p.shape[-1] == var_size for p, var_size in zip(particles, self.var_sizes) if var_size != 1)
        if self.dist_class not in self.dict_2torch_event.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2torch_event[self.dist_class](particles)

    def event2pred_event(self, particles):
        """
            Translate event particles from the format understandable by PyTorch to the format understandable by
                Predicate
            Take as input a single torch tensor.
            Return a tuple of torch tensors, each corresponding to one RV's value assignment
        """
        assert isinstance(particles, torch.Tensor)
        if self.dist_class not in self.dict_2pred_event.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2pred_event[self.dist_class](particles)

    def param2torch_param(self, params):
        """
            Translate parameters from the format understandable by Predicate to the format understandable by
                PyTorch
        """
        assert isinstance(params, torch.Tensor)
        if self.dist_class not in self.dict_2torch_param.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2torch_param[self.dist_class](params)

    def param2pred_event(self, params):
        """
            Translate parameters from the format understandable by PyTorch to the format understandable by
                Predicate
        """
        assert isinstance(params, torch.Tensor)
        if self.dist_class not in self.dict_2pred_param.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2pred_param[self.dist_class](params)

    """
        Default translation. 
            - event translation from pred to torch:
                Take a tuple of tensors. Concatenate along last dimension
            - event translation from torch to pred:
                Take a tensor. Split last dimension according to the sizes of RVs. Return a tuple of tensors
            - parameter translation from pred to torch:
                Assume same format. No special computation
            - parameter translation from torch to pred:
                Assume same format. No special computation
    """
    def _default_2torch_event(self, particles):
        result = torch.cat(particles, dim=-1)
        return result

    def _default_2pred_event(self, particles):
        result = torch.split(particles, self.var_sizes, dim=-1)
        return result

    def _default_do_nothing(self, stuff):
        return stuff

    """
        Categorical distribution. Assumes all RV have size 1
            - format check
                - var_sizes are all 1
                - var_constraints are all integer_interval
            - event translation from pred to torch:
                Take a tuple of tensors each corresponding to one RV's value assignment. Compute value by taking volume 
                product
            - event translation from torch to pred:
                Take volume modulo of the event values. Return a tuple a tensors
            - parameter translation from pred to torch:
                Flatten R.V. dimension into single dimension
            - parameter translation from torch to pred:
                Reshape last dimension into multiple dimensions, numbers of dims equal to the numbers of R.V.s
    """
    def _categorical_format_check(self):
        if not all(var_size == 1 for var_size in self.var_sizes):
            raise ValueError("Categorical distribution: Random Variables must all have size of 1. Found: '{}'"
                             .format(self.var_sizes))
        if not all(isinstance(c, torch.distributions.constraints.integer_interval) for c in self.var_constraints):
            raise ValueError("Categorical distribution: Random Variables must all declare 'integer_interval' value "
                             "constraints. Found: '{}'".format(self.var_constraints))

    def _categorical_var_span(self):
        # Helper function to determine the value range of each rv
        assert all(isinstance(c, integer_interval) for c in self.var_constraints)
        return list(c.upper_bound - c.lower_bound + 1 for c in self.var_constraints)

    def _categorical_2torch_event(self, particles):
        var_span = self._categorical_var_span()

        # Taking volume product
        volume_prod = 1
        base = 1
        # Going backward through spans to take product
        for val, span in zip(reversed(particles), reversed(var_span)):
            # Cumulative summation by the product of i_th variable's value with its base
            volume_prod += val * base
            # Base of i_th var is the product of the spans of all later variables (i.e. from (i+1)th to n_th variable)
            base *= span

        return volume_prod

    def _categorical_2pred_event(self, particles):
        assert isinstance(particles, torch.Tensor)
        var_span = self._categorical_var_span()

        # Treat values as volume products and take mod w.r.t. variables' spans
        particle_list = []
        residue = particles
        base = math.prod(var_span)
        # Going forward through spans to take modulo
        for span in var_span:
            base /= span
            particle_list.append(residue % base)
            residue = residue // base

        # Return a tuple of tensors
        result = tuple(particle_list)
        return result

    def _categorical_2torch_param(self, params):
        assert isinstance(params, torch.Tensor)
        var_span = self._categorical_var_span()
        old_shape = params.shape
        assert old_shape[-len(var_span):] == torch.Size(var_span)

        # Take a view of the parameter that flattens R.V. dimension
        new_shape = old_shape[:-len(var_span)] + torch.Size([-1])
        new_params = params.view(new_shape)

        return new_params

    def _categorical_2pred_param(self, params):
        assert isinstance(params, torch.Tensor)
        var_span = self._categorical_var_span()
        assert params.shape[-1] == math.prod(var_span)

        # Take a view of the parameter that expands last dimension into full R.V. dimensions
        new_shape = params.shape[:-1] + torch.Size(var_span)
        new_params = params.view(new_shape)

        return new_params



