"""
    Utility functions
"""
from collections.abc import Iterable
import numpy as np
import torch
import torch.distributions
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint, integer_interval
from torch.distributions.kl import kl_divergence


def intern_name(name: str, struc_type: str):
    """
        Add prefix and brackets to transform user provided structure name to internal name
    :param name:    Structure name
    :param struc_type:    one of "type", "predicate", or "conditional
    :return:        processed name
    """
    assert struc_type in ["type", "predicate", "conditional"], "unknown type for processing structure name"
    assert isinstance(name, str)
    if struc_type == "type":
        return "TYPE_[" + name.upper() + "]"
    elif struc_type == "predicate":
        return "PRED_[" + name.upper() + "]"
    else:
        return "COND_[" + name.upper() + "]"


def extern_name(name: str, struc_type: str):
    """
        Inverse operation of intern_name
    """
    assert struc_type in ["type", "predicate", "conditional"], "unknown type for processing structure name"
    assert isinstance(name, str)
    if struc_type == "type":
        assert name.find("TYPE_") >= 0
    if struc_type == "predicate":
        assert name.find("PRED_[") >= 0
    if struc_type == "conditional":
        assert name.find("COND_[") >= 0
    return name[6:-1]


def compatible_shape(msg_shape1, msg_shape2):
    """Checks whether the two given message shapes are compatible.

    Both `msg_shape1` and `msg_shape2` should be an iterable of `torch.Size` and have the contents
    ``(batch_shape, param_shape, sample_shape, event_shape)``. An empty shape, i.e., ``torch.Size([])``, will be deemed
    compatible with any other shape. `msg_shape1` is compatible with `msg_shape2` if and only if all of its four entries
    are compatible with their counterpart in `msg_shape2`.

    Parameters
    ----------
    msg_shape1 : tuple of torch.Size
        First shape. Should have the format ``(batch_shape, param_shape, sample_shape, event_shape)``.
    msg_shape2 : tuple of torch.Size
        Second shape. Same as `msg_shape1`.

    Returns
    -------
    bool
        True if both shape are compatible.
    """
    assert isinstance(msg_shape1, tuple) and len(msg_shape1) == 4 and all(isinstance(s, torch.Size) for s in msg_shape1)
    assert isinstance(msg_shape2, tuple) and len(msg_shape2) == 4 and all(isinstance(s, torch.Size) for s in msg_shape2)

    return all(s1 == torch.Size([]) or s2 == torch.Size([]) or s1 == s2 for s1, s2 in zip(msg_shape1, msg_shape2))


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
    def param2dist(cls, dist_class, param, b_shape=None, e_shape=None, dist_info=None):
        """Converts distribution parameter to a distribution instance.

        Depending on the context and Predicate knowledge format, the parameter `param` may belong to different
        representation systems, in which case it should be interpreted differently. Such specification should be
        sufficiently described in the argument `dist_info` in a prior consent format.

        The optional arguments `b_shape` and `e_shape` stand for distribution's batch shape and event shape
        respectively. They are used primarily for sanity check. Note that this event shape `e_shape` pertains to
        PyTorch's distribution class specification, and therefore may or may not be different from the event shape of
        particles in PySigma's cognitive format.

        Parameters
        ----------
        dist_class : type
            The distribution class. Must be a subclass of ``torch.distributions.Distribution``.
        param : torch.Tensor
            The parameter tensor. The last dimension is assumed to be the parameter dimension, and sizes of the
            dimensions at the front should be equal to `b_shape`.
        b_shape : torch.Size, optional
            The batch shape of the distribution. Used for shape sanity check.
        e_shape : torch.Size, optional
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
        assert b_shape is None or isinstance(b_shape, torch.Size)
        assert e_shape is None or isinstance(e_shape, torch.Size)
        assert dist_info is None or isinstance(dist_info, dict)

        if dist_class not in cls.dict_param2dist.keys():
            raise NotImplementedError("Conversion from parameters to distribution instance for distribution class '{}' "
                                      "not yet implemented".format(dist_class))

        dist = cls.dict_param2dist[dist_class](param, dist_info)
        if (b_shape is not None and dist.batch_shape != b_shape) or \
                (e_shape is not None and dist.event_shape != e_shape):
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
    def draw_particles(cls, dist, num_particles, dist_info=None):
        """Draw a list of `num_particles` event particles from the given distribution specified by `dist`. The event
        particles drawn will be in the format compatible with DistributionServer and PyTorch.

        Parameters
        ----------
        dist : torch.distributions.Distribution
            The distribution instance from which to sample particles
        num_particles : int
            The number of particles to be drawn
        dist_info : dict, optional
            Additional dist info necessary for drawing particles in the correct format.

        Returns
        -------
        torch.Tensor
            the list of particles drawn, of shape ``[num_particles, event_size]``

        Raises
        ------
        NotImplementedError
            If the given distribution yields multi-dimensional events, and no corresponding special drawing method is
            found in ``cls.dict_draw_particles`` method map.

        Notes
        -----
        Unless distribution-class-specific drawing method is specified and registered in ``cls.dict_draw_particles``
        method map, the distribution instance `dist` will be directly queried to draw the list of samples.

        The distribution instance `dist` is assumed batched, with a variable batch size(shape). However, we want to draw
        a single unique list of particles that is representative of each and every single distribution in the batch,
        i.e., draw a list of particles from the joint distribution regardless of the batch dimensions. Therefore, we
        take the view that drawing samples from `dist` simultaneously across the batch, which results in a sample tensor
        that involves the batch dimension, and ignoring the batch dimensions, is equivalent to first selecting uniformly
        which single distribution in the batch we wish draw from, and drawing samples from it, and repeating this
        process over and over again. The latter approach, when the samples are aggregated, yields a particle list
        that is representative of the joint distribution of the whole batch.

        Accordingly, the sampling process is implemented by drawing `n` batched samples from `dist`, where
        ``n = num_particles // batch_size + 1``, collapses the batch dimensions, random shuffle across the collapsed
        sample dimension, and truncate to select only a number of ``num_particles`` samples.
        """
        assert isinstance(dist, Distribution)
        assert isinstance(num_particles, int)
        assert dist_info is None or isinstance(dist_info, dict)

        dist_class = type(dist)
        if dist_class in cls.dict_draw_particles.keys():
            particles = cls.dict_draw_particles[dist_class](dist, num_particles, dist_info)
            assert particles.shape[0] == num_particles
        else:
            if len(dist.event_shape) > 1:
                raise NotImplementedError("Default particle drawing procedure only supports distribution class with "
                                          "1-dimensional events. For distribution class that yields multi-dimensional "
                                          "events, special method needs to be implemented and registered in "
                                          "`DistributionServer.dict_draw_particles` method map. Found distribution "
                                          "of type {} with event shape {}".format(dist_class, dist.batch_shape))
            batch_size = dist.batch_shape.numel()
            n = num_particles // batch_size + 1
            btch_ptcl = dist.sample_n(n)        # Draw n batched particles
            assert btch_ptcl.shape == torch.Size([n]) + dist.batch_shape + dist.event_shape
            flat_ptcl = btch_ptcl.view(-1, dist.event_shape[0])
            shuf_ptcl = flat_ptcl[torch.randperm(flat_ptcl.shape[0])]       # shuffle across collapsed sample dimension
            particles = torch.narrow(shuf_ptcl, dim=0, start=0, length=num_particles)   # truncate

        return particles

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
            A tensor with shape ``(sample_shape + [event_size])``

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
    def transform_param(cls, param, dist_info, trans):
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

    The architecture should hold one KnowledgeServer instance for each Predicate instantiated to cache knowledge
    contents and provide distribution related service.

    Parameters
    ----------
    dist_class : type
        The distribution class of the Predicate's knowledge. Must be a subclass of ``torch.distributions.Distribution``.
    rv_sizes : iterable of int
        The sizes of the random variables of the Predicate's knowledge. Note that the order given by the iterable will
        be respected.
    rv_constraints : iterable of torch.distributions.constraints.Constraint
        The value constraints of the random variables. Note that the order given by the iterable will be respected.
    rv_num_particles : iterable of int
        The number of marginal particles that should be drawn w.r.t. each random variable. Must have the same length as
        `rv_sizes` and `rv_constraints`, i.e., the number of random variables.
    dist_info : dict, optional
        An optional attribute dict that contains all necessary information for DistributionServer to draw particles and
        query particles' log pdf.

    Attributes
    ----------
    dist_class : type
    rv_sizes : tuple of int
    rv_constraints : tuple of torch.distributions.constraints.Constraint
    rv_num_particles : tuple of int
    dist_info : dict
    num_rvs : int
        Number of random variables involved in specifying the Predicate knowledge.
    e_shape : torch.Size
        The event shape of predicate's knowledge. Inferred form `rv_sizes`.
    particles : tuple of torch.Tensor
        The cached tuple of marginal particle event tensors corresponding to the random variables. This attribute is set
        when `draw_grid_particles` is called with ``update_cache=True``.
    log_densities : tuple of torch.Tensor
        The cached tuple of log sampling density tensors corresponding to each of the marginal particle event. This
        attribute is set when `draw_grid_particles` is called with ``update_cache=True``.

    Notes
    -----
    In order to provide service to both predicate nodes and conditional nodes in all stages, KnowledgeServer should
    store and manipulate data regarding the random variables only. In other words, only message components that do not
    involve batch dimensions should be cached; this includes particle value tensors and log sampling density tensors,
    but excludes both parameter and weight tensors. The latter ones' shapes are not invariant throughout the stages
    in the conditional subgraph, and therefore should be specified by the callee.

    Signatures for special private distribution class dependent methods:

    * Cognitive to PyTorch event format translation method: ``2torch_event(particles) --> particles``
    * PyTorch to Cognitive event format translation method: ``2cognitive_event(particles) --> particles``
    * Special marginal particle list sampling method: ``special_draw(batched_dist) --> particles, log_densities``
    """
    def __init__(self, dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info=None):
        assert issubclass(dist_class, Distribution)
        assert isinstance(rv_sizes, Iterable) and all(isinstance(s, int) and s > 0 for s in rv_sizes)
        assert isinstance(rv_constraints, Iterable) and all(isinstance(c, Constraint) for c in rv_constraints)
        assert isinstance(rv_num_particles, Iterable) and all(isinstance(n, int) and n > 0 for n in rv_num_particles)
        assert dist_info is None or isinstance(dist_info, dict)

        self.dist_class = dist_class
        self.rv_sizes = tuple(rv_sizes)
        self.rv_constraints = tuple(rv_constraints)
        self.rv_num_particles = tuple(rv_num_particles)
        self.dist_info = dist_info

        assert len(self.rv_sizes) == len(self.rv_constraints)
        self.num_rvs = len(self.rv_sizes)
        self.e_shape = torch.Size(rv_sizes)
        self.s_shape = torch.Size([n for n in self.rv_num_particles])

        # Cache
        self.particles = None
        self.log_densities = None

        # distribution-dependent translation method pointer. Indexed by distribution class
        self.dict_2torch_event = {
            torch.distributions.Categorical: self._categorical_2torch_event
        }
        self.dict_2cognitive_event = {
            torch.distributions.Categorical: self._categorical_2cognitive_event
        }
        self.dict_2special_draw = {
            tuple([integer_interval]): self._categorical_draw
        }

    """
    Public API
    """
    def draw_particles(self, batched_param, batch_shape, update_cache=True):
        """Draws new particles for the associated predicate w.r.t. the given `batched_param`. Returns necessary
        components to instantiate a particles message.

        This method is typically called by the predicate's LTMFN node during modification phase, in which the new
        updated batched parameter tensor has been obtained and provided by `batched_param`. This method is then
        proceed to:

        1. instantiate the batched distribution instances from the batched parameter tensor,
        2. draw a single unique list of **marginal** particle values w.r.t. each random variable from the entire batch
           of distribution instances,
        3. calculate their corresponding marginal sampling densities,

        Parameters
        ----------
        batched_param : torch.Tensor
            The new batched parameter tensor, of shape ``(batch_shape + [param_size])``.
        batch_shape : torch.Size
            The batch shape
        update_cache : bool
            Whether to replace the cache content in ``self.particles`` and ``self.log_densities`` with the result of
            calling this method.

        Returns
        -------
        particles : tuple of torch.Tensor
            The marginal particle lists w.r.t. each random variable in order.
        log_densities : tuple of torch.Tensor
            The marginal sampling log densities w.r.t. each random variable in order.

        Notes
        -----
        Some remarks regarding the aforementioned step 2 and 3:

        The tuple set of the types of the rv constraints specified in ``self.rv_constraints`` will be used to look up
        the pre-specified method map ``self.dict_2special_draw``. If an entry present, will used that method to obtain
        the returning ``particles`` and ``log_densities``. This is particularly useful, for instance, in the case of
        finite discrete random variables where a regular lattice should be drawn uniformly.

        Otherwise, the standard procedure will be carried out.
        ``
        """
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(batched_param, torch.Tensor) and batched_param.shape[:-1] == batch_shape
        assert isinstance(update_cache, bool)

        batched_dist = DistributionServer.param2dist(self.dist_class, batched_param, batch_shape, self.e_shape,
                                                     self.dist_info)

        # Look up for special draw method
        cstr = tuple(set(type(c) for c in self.rv_constraints))      # Take set to eliminate duplicate constraint types
        if cstr in self.dict_2special_draw.keys():
            particles, log_densities = self.dict_2special_draw[cstr](batched_dist)
        else:
            particles, log_densities = self._default_draw(batched_dist)
        # Check shape and type
        assert isinstance(particles, tuple) and \
            all(isinstance(p, torch.Tensor) and p.shape == torch.Size([self.s_shape[j], self.e_shape[j]])
                for j, p in enumerate(particles))
        assert isinstance(log_densities, tuple) and \
            all(isinstance(d, torch.Tensor) and d.shape == torch.Size(self.s_shape[j])
                for j, d in enumerate(log_densities))

        # Cache the particle list if asked for
        if update_cache:
            self.particles, self.log_densities = particles, log_densities

        return particles, log_densities

    def surrogate_log_prob(self, param, alt_particles=None, index_map=None):
        """Query the log pdf of the surrogate particles specified by `alt_particles` w.r.t. the cached distribution
        instance.

        A batched distribution instance will be instantiated from `param`, along with registered metadata in
        `self.dist_info`.

        If `index_map` is not specified, each entries in the iterable `alt_particles` must represent events of the
        Predicate's random argument at the same index in the predicate argument list. If the entry is 'None', the
        cached particle tensor of that predicate argument will be used instead.

        Alternatively, one can specify a dictionary `index_map` mapping integer index to an integer index or a list of
        indices. The entry ``alt_particles[i]`` will be taken as the particle tensor for the ``index_map[i]`` th
        predicate argument. If ``index_map[i]`` is a list of integers, then the particle tensor at this position will be
        interpreted as the concatenated/joint events of those predicate arguments whose indices are in ``index_map[i]``.
        Note that the entry ``alt_particles[i]`` can be ``None``, however in this case ``index_map[i]`` must refer to
        one predicate argument only. **If there is any predicate argument that is not referenced by values of**
        `index_map` **, then the returning** `surrogate_log_prob` **will be marginalized over this predicate argument.**

        Correspondingly, if ``index_map`` is specified, then all indices in ``alt_particles`` must appear as keys.

        Parameters
        ----------
        param : torch.Tensor, optional
            The alternative parameter from which a surrogate distribution instance is to be instantiated and log prob
            being queried. Should have the same shape as the cached ``self.batched_param``.
        alt_particles : list of (torch.Tensor or None), or None
            The surrogate particles to be queried. If not None, each entry must either be None, so that the
            corresponding cached articles will be used instead, or a torch.Tensor, with a shape of length 2 and the last
            dimension size equal to the corresponding value in ``self.rv_sizes``. Defaults to None
        index_map: dict, or None
            The optional index mapping. If specified, all applicable indices into `alt_particles` must appear as keys
            in this dict. The ``i`` th entry in `alt_particles` will be taken as the surrogate particles for the
            predicate argument whose index is ``index_map[i]`` if ``index_map[i]`` is an integer, or the joint surrogate
            particles for those arguments whose indices appear in ``index_map[i]`` if ``index_map[i]`` is a list.

        Returns
        -------
        torch.Tensor
            The log probability tensor, of shape ``(batch_shape + sample_shape)``, where ``batch_shape`` is the batch
            shape of the Predicate's knowledge, inferred from the shape of `param`, and ``sample_shape`` is the list of
            sample sizes of the queried particles in the order given by `alt_particles`. In other words,
            ``sample_shape[i] == alt_particles[i].shape[0]``. If ``alt_particles[i]``, it is the sample size of the
            cached particle tensor of the corresponding predicate argument.

        Raises
        ------
        AssertionError
            If `self.batched_param` is ``None``, meaning no cached parameters to instantiate distribution instance.
        AssertionError
            If `alt_particles` contains ``None`` but ``self.particles`` is also None, meaning no cached particles.
        AssertionError
            If `alt_param` is specified but it has different shape than ``self.batched_param``.
        """
        assert isinstance(param, torch.Tensor)
        assert alt_particles is None or isinstance(alt_particles, list)
        assert index_map is None or isinstance(index_map, dict)
        index_map_values = []
        if alt_particles is not None:
            if index_map is not None:
                # Check keys value set
                assert set(index_map.keys()) == set(range(len(alt_particles)))
                # Check values format
                assert all((isinstance(v, int) and v in range(self.num_rvs)) or
                           (isinstance(v, list) and all(isinstance(i, int) and i in range(self.num_rvs) for i in v))
                           for v in index_map.values())
                # Check values set
                for v in index_map.values():
                    if isinstance(v, int):
                        index_map_values.append(v)
                    else:
                        index_map_values += v
                assert set(index_map_values).issubset(set(range(self.num_rvs)))
                # Check None entry correspondence
                for i, p in enumerate(alt_particles):
                    if p is None:
                        assert isinstance(index_map[i], int)
                # Check tensor shape
                for k, v in index_map.items():
                    if isinstance(alt_particles[k], torch.Tensor):
                        assert alt_particles[k].dim() == 2
                        if isinstance(v, int):
                            assert alt_particles[k].shape[-1] == self.rv_sizes[v]
                        else:
                            assert alt_particles[k].shape[-1] == sum(self.rv_sizes[i] for i in v)
            else:
                # Check number of tensors and tensor format
                assert len(alt_particles) == self.num_rvs
                all(p is None or (isinstance(p, torch.Tensor) and p.dim() == 2 and p.shape[-1] == self.rv_sizes[i])
                    for i, p in enumerate(alt_particles))
        else:
            assert index_map is None
            alt_particles = (None,) * self.num_rvs

        assert all(isinstance(p, torch.Tensor) for p in alt_particles) or self.particles is not None, \
            "Found `None` in `alt_particles`, but no particles have been cached yet to be used instead."

        # Combinatorially concatenate particles to obtain full joint particle list.
        # This excludes the concatenated joint particles, if index_map is specified.
        num_unref = 0
        if index_map is None:
            precat_ptcl = list(p if p is not None else self.particles[i] for i, p in enumerate(alt_particles))
            cog_ptcl = KnowledgeServer.combinatorial_cat(precat_ptcl)
        else:
            unref_arg_ids = [v for v in range(self.num_rvs) if v not in index_map_values]
            num_unref = len(unref_arg_ids)
            # Directly perform combinatorial concatenation on provided list of particles, with unreferenced arguments'
            # particles appended to the front, if there's any. Resulting tensor have correct sample shape for the output
            # (except those corresponding to unreferenced arguments at the front) but wrong event shape
            # (and values if there's any unreferenced argument).
            ref_ptcl = list(p if p is not None else self.particles[index_map[i]] for i, p in enumerate(alt_particles))
            unref_ptcl = list(self.particles[i] for i in unref_arg_ids)
            precat_ptcl = unref_ptcl + ref_ptcl
            cat_ptcl = KnowledgeServer.combinatorial_cat(precat_ptcl)
            # So we split into chunks, permute and re-concatenate chunks so that the event dimension aligns with
            # predicate arguments in the correct order
            split_sizes = [self.rv_sizes[i] for i in unref_arg_ids] + \
                          [self.rv_sizes[index_map[i]] if isinstance(index_map[i], int)
                           else sum(self.rv_sizes[u] for u in index_map[i])
                           for i in range(len(alt_particles))]
            split_ptcl = torch.split(cat_ptcl, split_sizes, dim=-1)

            # Fill list
            cog_ptcl_list = [None] * self.num_rvs
            # Fill referenced particles
            for i in range(len(alt_particles)):
                v = index_map[i]
                if isinstance(v, int):
                    cog_ptcl_list[v] = split_ptcl[num_unref + i]
                else:
                    # Split joint particles again, and put particles in slot in the order given by the list
                    split_joint_sizes = [self.rv_sizes[u] for u in v]
                    split_joint_ptcl = torch.split(split_ptcl[num_unref + i], split_joint_sizes, dim=-1)
                    for j, u in enumerate(v):
                        cog_ptcl_list[u] = split_joint_ptcl[j]
            # Fill unreferenced particles
            for i, v in enumerate(unref_arg_ids):
                cog_ptcl_list[v] = split_ptcl[i]

            # Concatenate into original cognitive format
            cog_ptcl = torch.cat(cog_ptcl_list, dim=-1)
        assert cog_ptcl.shape[-1] == self.e_shape

        # Transform joint event values from Cognitive format to PyTorch format
        torch_ptcl = self.event2torch_event(cog_ptcl)
        # Instantiate the distribution instance
        dist = DistributionServer.param2dist(self.dist_class, param, self.dist_info)
        # Query DistributionServer
        log_prob = DistributionServer.log_prob(dist, torch_ptcl)
        # Marginalize over unreferenced dimensions if needed
        if num_unref > 0:
            # Marginalize first few dimensions, since we've put them at front
            prob = torch.exp(log_prob)
            marg_prob = torch.sum(prob, dim=tuple(range(num_unref)))
            log_prob = torch.log(marg_prob)

        return log_prob

    def event2torch_event(self, cat_particles):
        """Translates joint particle event values from the Cognitive format to a format understandable by PyTorch
        distribution class.

        Parameters
        ----------
        cat_particles : torch.Tensor
            A tensor representing the list of concatenated particle events in Cognitive format. Its last dimension will
            be taken as the event dimension and should be equal to the sum of rv sizes in ``self.rv_sizes``, while all
            other dimensions will be taken as the sample dimensions.

        Returns
        -------
        torch.Tensor
            A tensor representing a list of translated particle events from `cat_particles`. Its last dimension size
            depends on the PyTorch format representation of events, while the sizes of other dimensions are the same as
            `cat_particles`.

        Notes
        -----
        The specific translation method may vary depending on the distribution class. Therefore, this method serves only
        as an API entry point where the specific translation procedure will be looked up in ``self.dict_2torch_event``
        using the registered distribution class ``self.dist_class``. If no entry is found, then will assume no special
        translation is necessary and will return the input `cat_particles` as is.
        """
        assert isinstance(cat_particles, torch.Tensor) and cat_particles.shape[-1] == sum(self.rv_sizes)

        assert self.dist_class is not None, \
            "No distribution class has been registered. No way to translate given particle event values."

        if self.dist_class in self.dict_2torch_event.keys():
            result = self.dict_2torch_event[self.dist_class](cat_particles)
        else:
            result = cat_particles

        assert result.shape[:-1] == cat_particles.shape[:-1]
        return result

    def event2cognitive_event(self, particles):
        """Translates joint particle event values from the PyTorch distribution class format to Cognitive format.

        Parameters
        ----------
        particles : torch.Tensor
            A tensor representing the particle events in PyTorch-compatible format. Its last dimension will be taken
            as the event dimension, while all other dimensions will be taken as the sample dimensions.

        Returns
        -------
        torch.Tensor
            A concatenated tensor representing a list of translated particle events from `cat_particles`, where the
            events are concatenated along the last dimension, with size of each chunk in accordance with
            `self.rv_sizes`, and the sizes of all other dimensions are the same as `particles`.

        Notes
        -----
        The specific translation method may vary depending on the distribution class. Therefore, this method serves only
        as an API entry point where the specific translation procedure will be looked up in
        ``self.dict_2cognitive_event`` using the registered distribution class ``self.dist_class``. If no entry is
        found, then will assume no special translation is necessary and will return the input `cat_particles` as is.
        """
        assert isinstance(particles, torch.Tensor)

        assert self.dist_class is not None, \
            "No distribution class has been registered. No way to translate given particle event values."

        if self.dist_class in self.dict_2cognitive_event.keys():
            result = self.dict_2cognitive_event[self.dist_class](particles)
        else:
            result = particles

        assert result.shape[:-1] == particles.shape[:-1] and result.shape[-1] == sum(self.rv_sizes)
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

    @staticmethod
    def combinatorial_decat(cat_particles, split_sizes):
        """Helper static method that combinatorially de-concatenate the joint particles specified by `cat_particles`,
        with the event size of the particle tensors in each de-concatenated list given by `split_sizes`. This method
        implements the exact opposite operation of `combinatorial_cat`.

        An exception will be raised if `cat_particles` cannot be properly de-concatenated, for instance if it is not
        previously a result produced by `combinatorial_cat`.

        Parameters
        ----------
        cat_particles : torch.Tensor
            The concatenated particle tensor, of shape::

                [sample_size[0], ..., sample_size[m], sum(split_sizes)]

            where `m` is the number of variables to split/de-concatenate, to which the length of `split_sizes` should
            equal.
        split_sizes : list of int
            A list of integers denoting the event size of each split variable in order.

        Returns
        -------
        tuple of torch.Tensor
            The tuple of de-concatenated particles. The `i` th entry has shape ``[sample_size[i], split_sizes[i]]``.

        Raises
        ------
        ValueError
            If `cat_particles` was not a result from `combinatorial_cat` and cannot be properly de-concatenated.
        """
        assert isinstance(cat_particles, torch.Tensor) and cat_particles.dim() >= 2
        assert isinstance(split_sizes, list) and all(isinstance(s, int) and s > 0 for s in split_sizes)
        m = len(split_sizes)
        if m != cat_particles.dim() - 1:
            raise ValueError("Found {} numbers in `split_sizes`, therefore expect `cat_particles` have {} dimensions. "
                             "Instead found {} dimensions."
                             .format(m, m + 1, cat_particles.dim()))
        if sum(split_sizes) != cat_particles.shape[-1]:
            raise ValueError("The sum of `split_sizes` is {}, therefore expect the last dimension of `cat_particles`, "
                             "corresponding to the event dimension of joint particles, have an equal size. Instead "
                             "found a dimension size of {}"
                             .format(sum(split_sizes), cat_particles.shape[-1]))

        split_exp_ptcl = torch.split(cat_particles, split_sizes, dim=-1)
        split_ptcl = []
        # Check for uniqueness to determine if original tensor could be properly de-concatenated
        for i, exp_ptcl in enumerate(split_exp_ptcl):
            for j in range(m):
                if j != i:
                    exp_ptcl = exp_ptcl.unique(dim=j)
                    if exp_ptcl.shape[j] != 1:
                        raise ValueError("The provided `cat_particles` cannot be properly de-concatenated. The {}th "
                                         "split particle tensor with shape {}, cannot be reduced to unique elements "
                                         "along dimension {}.".format(i, split_exp_ptcl[i], j))
            split_ptcl.append(exp_ptcl.view(exp_ptcl.shape[i], exp_ptcl.shape[-1]))

        return tuple(split_ptcl)

    """
        Default methods that are distribution class independent
            - _default_draw:
                Draw a single unique list of marginal particles given batch of distributions and calculate marginal log 
                sampling densities .
    """
    def _default_draw(self, batched_dist):
        assert isinstance(batched_dist, Distribution)
        b_dims = len(batched_dist.batch_shape)
        b_size = batched_dist.batch_shape.numel()

        # Acquire raw joint particles in PyTorch format
        max_num_ptcl = max(self.rv_num_particles)
        raw_ptcl = DistributionServer.draw_particles(batched_dist, max_num_ptcl, self.dist_info)

        # Translate to cognitive format, split and adjust sample sizes
        joint_ptcl = self.event2cognitive_event(raw_ptcl)
        marg_ptcl_full = torch.split(joint_ptcl, self.rv_sizes, dim=-1)
        marg_ptcl_narrow = list(torch.narrow(p, dim=0, start=0, length=self.rv_num_particles[j])
                                for j, p in marg_ptcl_full)
        assert all(p.shape == torch.Size([self.s_shape[j], self.e_shape[j]]) for j, p in enumerate(marg_ptcl_narrow))

        # Obtain log densities w.r.t. the combinatorially concatenated event lattice
        comb_cat_ptcl = KnowledgeServer.combinatorial_cat(marg_ptcl_narrow)
        raw_comb_cat_ptcl = self.event2torch_event(comb_cat_ptcl)      # back to torch format again so DS can understand
        comb_log_dens = DistributionServer.log_prob(batched_dist, raw_comb_cat_ptcl)

        # Marginalize the lattice densities, by first taking average over batch dims then marginalize over individual
        # rv dims for each rv
        # Note that for the batch dims we take average, instead of marginalization by taking sum. This is because
        # lattice_dens is the CONDITIONAL probability of joint rv events conditioned on the batch index, NOT the joint
        # probability which involves the batch index as rv as well. We also assume the prior probability over the batch
        # index as rv is uniform.
        lattice_dens = torch.exp(comb_log_dens)
        for i in range(b_dims):
            lattice_dens = torch.sum(lattice_dens, dim=0)
        lattice_dens /= b_size     # IMPORTANT: normalize by the batch size, otherwise not valid probabilities
        assert lattice_dens.shape == self.s_shape

        marg_log_dens = []
        for j in range(self.num_rvs):
            marg_dens_j = lattice_dens
            for i in list(i for i in range(self.num_rvs) if i != j):
                marg_dens_j = torch.sum(marg_dens_j, dim=i, keepdim=True)
            marg_log_dens_j = torch.log(marg_dens_j.view(-1))
            marg_log_dens.append(marg_log_dens_j)
        assert all(d.shape == torch.Size(self.s_shape[j]) for j, d in enumerate(marg_log_dens))

        return tuple(marg_ptcl_narrow), tuple(marg_log_dens)

    """
        Categorical distribution. Assumes all RV have size 1
            - event translation from pred to torch:
                Take a tuple of tensors each corresponding to one RV's value assignment. Compute value by taking volume 
                product
            - event translation from torch to pred:
                Take volume modulo of the event values. Return a tuple a tensors
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
        base = np.prod(var_span)
        # Going forward through spans to take modulo
        for span in var_span:
            base /= span
            modulo_list.append(residue % base)
            residue = residue // base

        # Concatenate the modulo list
        result = torch.cat(modulo_list, dim=-1)
        return result

    def _categorical_draw(self, *args):
        """Draw particles components for Categorical distributions. Returns a finite discrete well-spaced lattice as the
        particle values and uniform log sampling densities.
        """
        var_span = self._categorical_var_span()
        particles = tuple(torch.arange(c.lower_bound, c.upper_bound + 1, 1.0) for c in self.rv_constraints)
        log_densities = tuple(torch.log(torch.ones(span) / span) for span in var_span)

        return particles, log_densities
