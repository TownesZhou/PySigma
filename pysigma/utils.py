"""
    Utility functions
"""
import torch
import torch.distributions
from torch.distributions import Distribution, ExponentialFamily
from torch.distributions.constraints import Constraint, integer_interval
from torch.distributions.kl import kl_divergence
from collections.abc import Iterable
import math


def intern_name(name: str, struc_type: str):
    """
        Add prefix and brackets to transform user provided structure name to internal name
    :param name:    Structure name
    :param struc_type:    one of "type", "predicate", or "conditional
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
    """Serving distribution class dependent utilities

    * Conversion between PyTorch distribution parameters and distribution instance:

      ``param2dist()``, ``dist2param()``

    * Translation between PyTorch distribution parameters and natural parameters for exponential family distribution:

      ``natural2exp_dist()``, ``exp_dist2natural()``

    * Get vector of moments from a given distribution instance:

      ``get_moments()``

    * Draw particles from distribution instance:

      ``draw_particles()``

    * Get log probability density from given particles:

      ``log_pdf()``


    Certain distribution classes require special handling, for example for those categorized as finite discrete,
    particle values will be drawn uniformly, covering every value in the RV's value range (support) once and only once,
    while assigning each particle its probability mass as its particle weight.

    Therefore we delegate all such special handling to this class on an individual basis.

    Note that input and output will conform to the format understandable by PyTorch's distribution class. To
    translate to and from formats compatible to PySigma's predicate knowledge, use KnowledgeServer class
    """
    """
    Public class method API
    """
    @classmethod
    def param2dist(cls, dist_class, param, b_shape, e_shape, dist_info=None):
        """Converts distribution parameter to a distribution instance.

        Depending on the context and Predicate knowledge format, the parameter `param` may belong to different
        representation systems, in which case it should be interpreted differently. Such specification should be
        sufficiently described in the argument `dist_info` in a prior consent format.

        The argument `b_shape` and `e_shape` stand for distribution's batch shape and event shape respectively. They
        are used primarily for sanity check. Note that this event shape `e_shape` pertains to PyTorch's distribution
        class specification, and therefore may or may not be different from the event shape of particles in PySigma's
        cognitive format.

        Parameters
        ----------
        dist_class : type
            The distribution class. Must be a subclass of ``torch.distributions.Distribution``.
        param : torch.Tensor
            The parameter tensor. The last dimension is assumed to be the parameter dimension, and sizes of the
            dimensions at the front should be equal to `b_shape`.
        b_shape : torch.Size
            The batch shape of the distribution. Used for shape sanity check.
        e_shape : torch.Size
            The presumed event shape of the distribution. Used for shape sanity check.
        dist_info : dict, optional
            An optional dict containing all relevant information in order to correctly interpret the parameter `param`.

        Returns
        -------
        torch.distributions.Distribution
            The instantiated distribution instance.

        Raises
        ------
        NotImplementedError
            If the conversion procedure specific to `dist_class` has not been implemented yet.
        ValueError
            If the converted distribution instance has different batch shape and event shape than specified `b_shape`
            and `e_shape` respectively.
        """
        assert issubclass(dist_class, torch.distributions.Distribution)
        assert isinstance(param, torch.Tensor)
        assert isinstance(b_shape, torch.Size)
        assert isinstance(e_shape, torch.Size)
        assert dist_info is None or isinstance(dist_info, dict)

        if dist_class not in cls.dict_param2dist.keys():
            raise NotImplementedError("Conversion from parameters to distribution instance for distribution class '{}' "
                                      "not yet implemented".format(dist_class))

        dist = cls.dict_param2dist[dist_class](param, dist_info)
        if dist.batch_shape != b_shape or dist.event_shape != e_shape:
            raise ValueError("The shape of the generated distribution {} does not match the provided shape. "
                             "Provided (batch_shape, event_shape) == ({}, {}), but instead got ({}, {})."
                             .format(str(dist), b_shape, e_shape, dist.batch_shape, dist.event_shape))
        return dist

    @classmethod
    def dist2param(cls, dist, dist_info=None):
        """Extract the parameter tensor from a given distribution instance.

        Depending on the context and Predicate knowledge format, the desired parameter may belong to different
        representation systems, in which case it should be generated differently. Such specification should be
        sufficiently described in the argument `dist_info` in a prior consent format.

        Parameters
        ----------
        dist : torch.distributions.Distribution
            The distribution instance from which to extract the parameter.
        dist_info : dict, optional
            An optional dict containing all relevant information in order to correctly generate the parameter `param`.

        Returns
        -------
        torch.Tensor
            The parameter tensor in the desired format.

        Raises
        ------
        NotImplementedError
            If the conversion procedure specific to the distribution class of `dist` has not been implemented yet.
        """
        assert isinstance(dist, Distribution)
        assert dist_info is None or isinstance(dist_info, dict)
        dist_class = type(dist)

        if dist_class not in cls.dict_dist2param.keys():
            raise NotImplementedError("Conversion from distribution instance to parameters for distribution class '{}' "
                                      "not yet implemented".format(dist_class))
        return cls.dict_dist2param[dist_class](dist, dist_info)

    @classmethod
    def get_moments(cls, dist, n_moments):
        """Get vector of moments from a given distribution instance

        .. todo::
           Implement with dist_info
        """
        assert isinstance(dist, Distribution)
        assert isinstance(n_moments, int) and n_moments > 0

        dist_class = type(dist)
        if dist_class not in cls.dict_get_moments.keys():
            raise NotImplementedError("Get moments method for distribution class '{}' not yet implemented"
                                      .format(dist_class))
        return cls.dict_get_moments[dist_class](dist, n_moments)

    @classmethod
    def draw_particles(cls, dist, num_particles, b_shape, e_shape):
        """
            .. todo::
               Gibbs sampling procedure
            Draw a given number of particles from the given batch of distribution instances. Return a tuple:
                    (particles, weights, sampling_log_densities)

            The Gibbs' Sampling procedure is used to draw a single list of particles that will be used by each
                distribution instance in the batch to derive individual weights by importance weighting.

            Particles drawn are in the format compatible with PyTorch's distribution class
        """
        assert isinstance(dist, Distribution)
        assert isinstance(num_particles, int)
        assert isinstance(b_shape, torch.Size)
        assert isinstance(e_shape, torch.Size)

        dist_class = type(dist)
        if dist_class not in cls.dict_draw_particles.keys():
            raise NotImplementedError("Draw particles method for distribution class '{}' not yet implemented"
                                      .format(dist_class))
        particles, weights, sampling_log_densities = cls.dict_draw_particles[dist_class](dist, num_particles)

        # shape check
        s_shape = torch.Size([num_particles])
        assert isinstance(particles, torch.Tensor) and particles.shape == s_shape + b_shape + e_shape
        assert (isinstance(weights, int) and weights == 1) or (isinstance(weights, torch.Tensor) and
                                                               weights.shape == s_shape + b_shape)
        assert isinstance(sampling_log_densities, torch.Tensor) and sampling_log_densities.shape == s_shape + b_shape

        return particles, weights, sampling_log_densities

    @classmethod
    def log_prob(cls, dist, values):
        """Get the log probability mass/density of the given particle values w.r.t. the given batched distribution
        instance.

        The particle value should be in PyTorch format that is compatible with PyTorch's distribution classes. This
        means the last dimension of `values` is assumed the event dimension, and should be compatible with, if not
        identical to, ``dist.event_shape``. Every other dimensions to the front is assumed sample dimensions, the sizes
        of which together forms the ``sample_shape``.

        The distribution instance `dist` is assumed batched. In other words, its batch shape ``dist.batch_shape`` should
        not be empty.

        Parameters
        ----------
        dist : torch.distributions.Distribution
            A batched distribution instance. Its batch shape ``dist.batch_shape`` should not be empty.
        values : torch.Tensor
            A 2D tensor with shape ``(sample_shape + [event_size])``

        Returns
        -------
        torch.Tensor
            The log probability mass/density tensor, of shape ``(dist.batch_shape + sample_shape)``

        Raises
        ------
        AssertionError
            If the ``event_size`` found in `values` is different from `dist.event_shape`.
        """
        assert isinstance(dist, Distribution) and len(dist.batch_shape) >= 1
        assert isinstance(values, torch.Tensor) and values.dim() >= 2

        # dist_class = type(dist)
        # if dist_class not in cls.dict_log_pdf.keys():
        #     raise NotImplementedError("Get log pdf method for distribution class '{}' not yet implemented"
        #                               .format(dist_class))
        # return cls.dict_log_pdf[dist_class](dist, particles)

        # Extract shapes
        sample_shape, event_shape = values.shape[:-1], values.shape[-1:]
        batch_shape = dist.batch_shape
        assert event_shape == dist.event_shape, \
            "The event shape ({}) found in the given particles is different from the event shape ({}) found in the " \
            "distribution instance".format(event_shape, dist.event_shape)

        # Insert singleton dimensions into the particles tensor, and repeat along those dimensions to expand to
        # full batch shape.
        for i in range(len(batch_shape)):
            values = values.unsqueeze(dim=len(sample_shape))
        repeat_times = [1] * len(sample_shape) + list(batch_shape) + [1]
        values = values.repeat(repeat_times)

        # Query the actual distribution instance
        log_prob = dist.log_prob(values)

        # The returned log prob tensor should have shape (sample_shape + batch_shape). We need to permute these
        # dimensions to conform to the Cognitive format
        assert log_prob.shape == sample_shape + batch_shape
        perm_order = [len(sample_shape) + i for i in range(len(batch_shape))] + [i for i in range(len(sample_shape))]
        log_prob = log_prob.permute(perm_order)

        # Also call contiguous() to rearrange the memory layout
        log_prob = log_prob.contiguous()

        return log_prob

    @classmethod
    def kl_norm(cls, dist1, dist2):
        """
            Get the norm of the KL divergence of two given batched distributions
        """
        assert isinstance(dist1, Distribution) and isinstance(dist2, Distribution)
        assert dist1.batch_shape == dist2.batch_shape
        kl = kl_divergence(dist1, dist2)
        kl_norm = kl.norm()
        return kl_norm

    @classmethod
    def transform_param(cls, param, dist_class, trans):
        """
            .. todo::
               To implement

            Return the parameter of the transformed distribution
        """
        pass

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
        """
            Default method for drawing particles. Draw according to the distribution itself.
            Therefore, weights are uniform, and sampling log densities are the distribution's log pdf
        """
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
    def _categorical_param2dist(cls, params, dist_info):
        """
            For categorical, params assumed to be fed as the 'probs' attribute
            # TODO: different parameter scheme and dist_info schema specification
        """
        dist = torch.distributions.Categorical(probs=params)
        return dist

    @classmethod
    def _categorical_dist2param(cls, dist, dist_info):
        """
            # TODO: different parameter scheme and dist_info schema specification
        """
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

        # Since we are effectively drawing particles uniformly from the finite discrete domain, the sampling pdf is also
        #   uniform
        sampling_log_densities = 0

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


class KnowledgeServer:
    """Knowledge Server class. Provides service regarding a Predicate's knowledge.

    Parameters
    ----------
    dist_class : type
        The distribution class of the Predicate's knowledge. Must be a subclass of ``torch.distributions.Distribution``.
    rv_sizes : iterable of ints
        The sizes of the random variables of the Predicate's knowledge. Note that the order given by the iterable will
        be respected.
    rv_constraints : iterable of torch.distributions.constraints.Constraint
        The value constraints of the random variables. Note that the order given by the iterable will be respected.
    dist_info : dict, optional
        An optional attribute dict that contains all necessary information for DistributionServer to draw particles and
        query particles' log pdf.

    Attributes
    ----------
    dist_class : type
    rv_sizes : tuple of ints
    rv_constraints : tuple of torch.distributions.constraints.Constraint
    dist_info : dict
    num_rvs : int
        Number of random variables involved in specifying the Predicate knowledge.
    batched_param : torch.Tensor
        The cached batched parameter tensor of the Predicate's knowledge. This attribute is set when
        `draw_grid_particles` is called with ``update_cache=True``.
    batched_dist : torch.distributions.Distribution
        The cached batched distribution instance that is instantiated from `batched_param`. This attribute is set when
        `draw_grid_particles` is called with ``update_cache=True``.
    particles : tuple of torch.Tensor
        The cached tuple of marginal particle event tensors corresponding to the random variables. This attribute is set
        when `draw_grid_particles` is called with ``update_cache=True``.
    log_densities : tuple of torch.Tensor
        The cached tuple of log sampling density tensors corresponding to each of the marginal particle event. This
        attribute is set when `draw_grid_particles` is called with ``update_cache=True``.
    """
    def __init__(self, dist_class, rv_sizes, rv_constraints, dist_info=None):
        assert issubclass(dist_class, Distribution)
        assert isinstance(rv_sizes, Iterable) and all(isinstance(s, int) and s > 0 for s in rv_sizes)
        assert isinstance(rv_constraints, Iterable) and all(isinstance(c, Constraint) for c in rv_constraints)
        assert dist_info is None or isinstance(dist_info, dict)

        self.dist_class = dist_class
        self.rv_sizes = tuple(rv_sizes)
        self.rv_constraints = tuple(rv_constraints)
        self.dist_info = dist_info

        assert len(self.rv_sizes) == len(self.rv_constraints)
        self.num_rvs = len(self.rv_sizes)

        # Cache
        self.batched_param = None
        self.batched_dist = None
        self.particles = None
        self.log_densities = None

        # distribution-dependent translation method pointer. Indexed by distribution class
        self.dict_2torch_event = {
            torch.distributions.Categorical: self._categorical_2torch_event
        }
        self.dict_2cognitive_event = {
            torch.distributions.Categorical: self._categorical_2cognitive_event
        }

    """
    Public API
    """
    def draw_grid_particles(self, new_param, update_cache=True):
        """
        .. todo::
           To implement
        """
        pass

    def surrogate_log_prob(self, surrogate_particles):
        """Query the log pdf of the surrogate particles specified by `surrogate_particles` w.r.t. the cached
        distribution instance.

        Each particle tensor in `surrogate_particles` must correspond to a Predicate's random variable in the given
        order. Alternatively, ``None`` can be specified as an entry in `surrogate_particles` so that the cached particle
        tensor of the corresponding RV remembered by this KnowledgeServer instance will be used instead.

        Parameters
        ----------
        surrogate_particles : list of torch.Tensor and/or None
            The surrogate particles to be queried. Each entry must either be None, so that the corresponding cached
            particles will be used instead, or a torch.Tensor, with a shape of length 2 and the last dimension size
            equal to the corresponding value in ``self.rv_sizes``.

        Returns
        -------
        torch.Tensor
            The log probability tensor, of shape ``(sample_shape + batch_shape)``, where ``sample_shape`` is the list
            of sample sizes of the queried particles in order (either those provided by `surrogate_particles` or those
            in ``self.particles``), and ``batch_shape`` is the batch shape of the Predicate's knowledge.

        Raises
        ------
        AssertionError
            If `self.batched_param` is ``None``, meaning no cached parameters to instantiate distribution instance.
        AssertionError
            If `surrogate_particles` contains ``None`` but ``self.particles`` is also None, meaning no cached particles.
        """
        assert isinstance(surrogate_particles, list) and len(surrogate_particles) == self.num_rvs and \
            all(p is None or (isinstance(p, torch.Tensor) and p.dim() == 2 and p.shape[1] == self.rv_sizes[i])
                for i, p in enumerate(surrogate_particles))

        assert self.batched_dist is not None, \
            "No distribution instance has been cached, so cannot look up particles' log pdf."
        assert all(isinstance(p, torch.Tensor) for p in surrogate_particles) or self.particles is not None, \
            "Found `None` in `surrogate_particles`, but no particles have been cached yet to be used instead."

        query_particles = list(p if p is not None else self.particles[i] for i, p in enumerate(surrogate_particles))

        # Combinatorially concatenate particles to obtain full joint particle list
        cat_particles = KnowledgeServer.combinatorial_cat(query_particles)
        # Transform joint event values from Cognitive format to PyTorch format
        torch_particles = self.event2torch_event(cat_particles)

        # Query DistributionServer
        log_prob = DistributionServer.log_prob(self.batched_dist, torch_particles)

        return log_prob

    def event2torch_event(self, cat_particles):
        """Translates joint particle event values from the Cognitive format to a format understandable by PyTorch
        distribution class.

        Parameters
        ----------
        cat_particles : torch.Tensor
            A 2D tensor representing the list of concatenated particle events in Cognitive format. Its 2nd dimension
            size should be equal to the sum of rv sizes in ``self.rv_sizes``.

        Returns
        -------
        torch.Tensor
            A 2D tensor representing a list of translated particle events from `cat_particles`. Its 1st dimension size
            is equal to the 1st dimension size of `cat_particles`.

        Notes
        -----
        The specific translation method may vary depending on the distribution class. Therefore, this method serves only
        as an API entry point where the specific translation procedure will be looked up in ``self.dict_2torch_event``
        using the registered distribution class ``self.dist_class``. If no entry is found, then will assume no special
        translation is necessary and will return the input `cat_particles` as is.
        """
        assert isinstance(cat_particles, torch.Tensor) and cat_particles.dim() == 2 and \
            cat_particles.shape[1] == sum(self.rv_sizes)

        assert self.dist_class is not None, \
            "No distribution class has been registered. No way to translate given particle event values."

        if self.dist_class in self.dict_2torch_event.keys():
            result = self.dict_2torch_event[self.dist_class](cat_particles)
        else:
            result = cat_particles

        assert result.shape[0] == cat_particles.shape[0]
        return result

    def event2cognitive_event(self, particles):
        """Translates joint particle event values from the PyTorch distribution class format to Cognitive format.

        Parameters
        ----------
        particles : torch.Tensor
            A 2D tensor representing the list of concatenated particle events in PyTorch format.

        Returns
        -------
        torch.Tensor
            A 2D tensor representing a list of translated particle events from `cat_particles`. Its 1st dimension size
            is equal to the 1st dimension size of `cat_particles`, and 2nd dimension size equal to the sum of rv sizes
            in ``self.rv_sizes``.

        Notes
        -----
        The specific translation method may vary depending on the distribution class. Therefore, this method serves only
        as an API entry point where the specific translation procedure will be looked up in
        ``self.dict_2cognitive_event`` using the registered distribution class ``self.dist_class``. If no entry is
        found, then will assume no special translation is necessary and will return the input `cat_particles` as is.
        """
        assert isinstance(particles, torch.Tensor) and particles.dim() == 2

        assert self.dist_class is not None, \
            "No distribution class has been registered. No way to translate given particle event values."

        if self.dist_class in self.dict_2cognitive_event.keys():
            result = self.dict_2cognitive_event[self.dist_class](particles)
        else:
            result = particles

        assert result.shape[0] == particles.shape[0] and result.shape[1] == sum(self.rv_sizes)
        return result

    """
    Utility static methods
    """

    @staticmethod
    def combinatorial_cat(particles):
        """Helper static method that combinatorially concatenates the list of event particles specified by `particles`.

        Returns the contained tensor directly if there is only one entry in `particles`.

        Parameters
        ----------
        particles : iterable of torch.Tensor
            The list of particles to be concatenated. Each element should be a tensor with a shape of length 2, where
            the first dimension is assumed the sample dimension, and second dimension assumed the event dimension.

        Returns
        -------
        torch.Tensor
            The combinatorially concatenated event particle tensor of shape::

                [sample_size[0], ..., sample_size[m], event_size[0]+...+event_size[m]]

            where ``sample_size[i]`` is the sample size of the ``i`` th particle, and similarly is ``event_size[i]``.
            Its total number of dimensions, i.e. ``.dim()``, is equal to the number of random variables plus 1.
        """
        assert isinstance(particles, Iterable) and len(list(particles)) > 0 and \
            all(isinstance(p, torch.Tensor) and p.dim() == 2 for p in particles)

        if len(particles) < 2:
            return particles[0]

        particles = list(particles)
        num = len(particles)
        sample_size_list = [p.shape[0] for p in particles]
        event_size = sum([e.shape[1] for e in particles])

        # 1. Repeat each particle tensor to expand to full sample dimensions.
        exp_particles = []
        for i, p in enumerate(particles):
            # Insert singleton sample dimensions
            dims = ([1] * (num - 1))        # singleton dimensions
            dims.insert(i, p.shape[0])      # original sample dimension
            dims.append(p.shape[1])         # append event dimension at the end
            p_viewed = p.view(dims)
            # Repeat tensor along inserted singleton dimensions
            repeat_times = sample_size_list[:i] + [1] + sample_size_list[i + 1:] + [1]
            p_repeated = p_viewed.repeat(repeat_times)
            exp_particles.append(p_repeated)

        # 2. Concatenate along the event dimensions
        comb_cat = torch.cat(exp_particles, dim=-1)
        assert comb_cat.shape == torch.Size(sample_size_list + [event_size])

        return comb_cat

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
    def _categorical_var_span(self):
        # Helper function to determine the value range of each rv
        assert all(isinstance(c, integer_interval) for c in self.rv_constraints)
        return tuple(c.upper_bound - c.lower_bound + 1 for c in self.rv_constraints)

    def _categorical_2torch_event(self, particles):
        assert all(s == 1 for s in self.rv_sizes), \
            "While attempting to translate Categorical events to PyTorch format, expect random variable sizes to be " \
            "all 1, but instead found sizes {}".format(self.rv_sizes)

        # First split joint event values
        split_particles = torch.split(particles, 1, dim=-1)
        # Get rv span
        var_span = self._categorical_var_span()
        # Taking volume product
        volume_prod = 1
        base = 1
        # Going backward through spans to take product
        for val, span in zip(reversed(split_particles), reversed(var_span)):
            # Cumulative summation by the product of i_th variable's value with its base
            volume_prod += val * base
            # Base of i_th var is the product of the spans of all later variables (i.e. from (i+1)th to n_th variable)
            base *= span

        return volume_prod

    def _categorical_2cognitive_event(self, particles):
        assert isinstance(particles, torch.Tensor)
        assert all(s == 1 for s in self.rv_sizes), \
            "While attempting to translate Categorical events to PyTorch format, expect random variable sizes to be " \
            "all 1, but instead found sizes {}".format(self.rv_sizes)

        var_span = self._categorical_var_span()
        # Treat values as volume products and take mod w.r.t. variables' spans
        modulo_list = []
        residue = particles
        base = math.prod(var_span)
        # Going forward through spans to take modulo
        for span in var_span:
            base /= span
            modulo_list.append(residue % base)
            residue = residue // base

        # Concatenate the modulo list
        result = torch.cat(modulo_list, dim=-1)
        return result
