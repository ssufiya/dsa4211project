"""step-size adaptation classes, currently tightly linked to CMA,
because `hsig` is computed in the base class
"""
from __future__ import absolute_import, division, print_function  #, unicode_literals, with_statement
import warnings as _warnings
import numpy as np
from numpy import square as _square, sqrt as _sqrt
from .utilities import utils
from .utilities.math import Mh
from . import warnings_and_exceptions as _cma_warnings
del absolute_import, division, print_function  #, unicode_literals, with_statement

csa_dampdown_fac = 1  # for the time being a module global variable
'''Asymmetric damping parameter read by ``CMAAdaptSigmaCSA` on instantiation and
   on reset in `.initialize_settings``'''

tpa_dampdown_fac = 1  # for the time being a module global variable
'''Asymmetric damping factor, read by TPA on instantation and reset.
   Is 4 in the restricted CMA of 2024, was tentatively 2,
   4 mostly does not change the standard performance much'''


_warnings.filterwarnings('once', message="Missing property ``path_.*")

def _norm(x):
    return np.sqrt(np.sum(np.square(x)))

class CMAAdaptSigmaBase(object):
    """step-size adaptation base class, implement `hsig` (for stalling
    distribution update) functionality via an isotropic evolution path.

    Details: `hsig` or `_update_ps` must be called before the sampling
    distribution is changed. `_update_ps` depends heavily on
    `cma.CMAEvolutionStrategy`.
    """
    def __init__(self, *args, **kwargs):
        self.is_initialized_base = False
        self._ps_updated_iteration = -1
        self.delta = 1
        "cumulated effect of adaptation"
    def initialize_base(self, es, reset=False):
        """set parameters and state variable based on dimension,
        mueff and possibly further options.

        Overwrites `is_initialized_base` and attributes with value `None`.
        """
        ## meta_parameters.cs_exponent == 1.0
        b = 1.0
        ## meta_parameters.cs_multiplier == 1.0
        if reset or not hasattr(self, 'cs') or self.cs is None:
            self.cs = 1.0 * (es.sp.weights.mueff + 2)**b / (es.N**b + (es.sp.weights.mueff + 3)**b)
        if reset or not hasattr(self, 'ps') or self.ps is None:
            self.ps = np.zeros(es.N)
        self.is_initialized_base = True
        return self
    def _update_ps(self, es):
        """update the isotropic evolution path.

        Using ``es`` attributes ``mean``, ``mean_old``, ``sigma``,
        ``sigma_vec``, ``sp.weights.mueff``, ``cp.cmean`` and
        ``sm.transform_inverse``.

        :type es: CMAEvolutionStrategy
        """
        if not self.is_initialized_base:
            self.initialize_base(es)
        if self._ps_updated_iteration == es.countiter:
            return
        cs = self.cs  # we could use a different cs?
        self.ps *= 1 - cs
        self.ps += (cs * (2 - cs))**0.5 * es.isotropic_mean_shift  # uses BD from before the C-update
        self._ps_updated_iteration = es.countiter
    def hsig(self, es):
        """return "OK-signal" for rank-one update.

        Return `True` (OK) or `False` ("stall" rank-one update), based on the
        length of an evolution path.
        """
        self._update_ps(es)
        # DONE?: the following is executed again in update2? hsig is not called in update2.
        #       Possible fix: save ps and use self._ps_updated_iteration == es.countiter
        #       or a hash es.pc to make the computation conditional?
        if es.opts.get('CSA_invariant_path'):  # was pc_for_ps = 'pc for ps' in es.opts['vv']
            # was: es.D**-1 * np.dot(es.B.T, es.pc)
            ps = es.sm.transform_inverse(es.pc)
            cs = es.sp.cc
        elif self.ps is None:
            return True
        else:
            ps = self.ps
            cs = self.cs
        squared_sum = np.sum(ps**2) / (1 - (1 - cs)**(2 * es.countiter))
        # correction with self.countiter seems not necessary,
        # as pc also starts with zero
        return squared_sum / es.N - 1 < 1 + 4. / (es.N + 1)

    def update(self, es, **kwargs):
        """change `es.sigma` in place, deprecated?"""
        es.sigma *= self.update2(es, **kwargs)

    def update2(self, es, **kwargs):
        """return sigma change factor and update self.delta.

        ``self.delta == sigma/sigma0`` accumulates all past changes
        starting from `1.0`.

        Unlike `update`, `update2` is not supposed to change attributes
        in `es`, specifically it should not change `es.sigma`.
        """
        self._update_ps(es)
        raise NotImplementedError('must be implemented in a derived class')

    def check_consistency(self, es):
        """make consistency checks with a `CMAEvolutionStrategy` instance
        as input
        """
class CMAAdaptSigmaNone(CMAAdaptSigmaBase):
    """constant step-size sigma"""
    def update(self, es, **kwargs):
        """no update, ``es.sigma`` remains constant.
        """
        pass
class CMAAdaptSigmaDistanceProportional(CMAAdaptSigmaBase):
    """artificial setting of ``sigma`` proportional to ||m||,

    specifically ``sigma = coefficient * mueff * norm(mean) / dimension / c_m``
    or ``sigma = coefficient * norm(mean)`` in direct mode.

    >>> import cma
    >>> cma.evolution_strategy._redistribute_sigma_above = False
    >>> es = cma.CMAEvolutionStrategy(4 * [1/2], 2, {'verbose': -9,
    ...         'AdaptSigma': cma.sigma_adaptation.CMAAdaptSigmaDistanceProportional(
    ...                           1.2, True)})
    >>> # now we can call es.optimize(...)
    >>> assert isinstance(es.adapt_sigma, cma.sigma_adaptation.CMAAdaptSigmaDistanceProportional
    ...                   ), es.adapt_sigma
    >>> assert es.sigma == 1.2, (es.adapt_sigma.__dict__, es.sigma)

    The optimal `coefficient` in infinite dimension is ``1.253 = (pi/2)**0.5``,
    the optimal mueff is ``lambda / pi``, hence the optimal phi is ``pi/2 x
    lambda / pi / 2 = lambda / 4`` where exp(-phi/n) is the (log-)expected
    convergence rate per iteration.

    This is mainly useful for test purposes, e.g. to simulate optimal progress
    rates.

    Details: setting `cma.options_parameters.CMAOptions._stationary_sphere` to
    `True` has on scaling invariant functions the "same effect". Instead of
    changing `sigma`, it resets ``norm(mean)`` at the end of `tell`.
    """
    def __init__(self, coefficient=1.2, direct_mode=False, **kwargs):
        """pass coefficient multiplier for normalized step-size"""
        super(CMAAdaptSigmaDistanceProportional, self).__init__() # base class provides method hsig()
        self.coefficient = coefficient
        self.direct_mode = direct_mode
        '''experimental: when True, interpret coefficient as sigma / norm(mean)'''
        self.is_initialized = False
    def initialize(self, es, *args, **kwargs):
        """same as update, set the correct step-size before the first iteration"""
        self.update(es, **kwargs)
        self.is_initialized = True
        return self
    def update(self, es, **kwargs):
        """update ``es.sigma`` by calling `update2`.
        """
        es.sigma *= self.update2(es)
    def update2(self, es, **kwargs):
        """return sigma update factor.

        Uses attributes ``.N``, ``.sp.weights.mueff``, ``.mean``, and
        ``.sp.cmean`` of input `es`.
        """
        factor = self.coefficient * _norm(es.mean) / es.sigma
        return factor if self.direct_mode else (
            factor * es.sp.weights.mueff / es.N / es.sp.cmean)

class CMAAdaptSigmaDistanceProportional2(CMAAdaptSigmaBase):
    """update ``es.sigma = self.sigma * _norm(es.mean)``,

    where ``self.sigma == es.sigma0 / _norm(x0) ``, for example for test
    purposes, e.g. to simulate optimal progress rates, where ``self.sigma ==
    es.sigma0``.

    >>> import cma, warnings
    >>> cma.evolution_strategy._redistribute_sigma_above = False
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', cma.warnings_and_exceptions.NeverTestedWarning)
    ...     es = cma.CMA(4 * [1/2], 2, {'verbose': -9,
    ...         'AdaptSigma': cma.sigma_adaptation.CMAAdaptSigmaDistanceProportional2})
    >>> assert isinstance(es.adapt_sigma, cma.sigma_adaptation.CMAAdaptSigmaDistanceProportional2
    ...                   ), es.adapt_sigma
    >>> assert es.adapt_sigma.sigma == 2, (es.adapt_sigma.__dict__, es.sigma0)
    >>> # now we can call es.optimize(...)

    CAVEAT: this class was never used.

    Details: ``.initialize`` is called in `CMAEvolutionStrategy.__init__`.

    `evolution_strategy._redistribute_sigma_above` changes the meaning
    of `es.sigma` and would brake this update when invoked. To be safe,
    assign::

        cma.evolution_strategy._redistribute_sigma_above = False

    before using `CMAAdaptSigmaDistanceProportional2`.
    """
    def __init__(self, **kwargs):
        """pass distance proportional step-size given ``||m|| = 1``"""
        super(CMAAdaptSigmaDistanceProportional2, self).__init__()  # base class provides method hsig()
        self.sigma = None
        self.is_initialized = False
        _warnings.warn("`CMAAdaptSigmaDistanceProportional2` was never thoroughly tested",
                       category=_cma_warnings.NeverTestedWarning)
    def initialize(self, es, *args, **kwargs):
        """set ``self.sigma = es.sigma0``."""
        self.norm0 = _norm(es.mean)
        self.sigma = es.sigma0 / self.norm0
        self.is_initialized = True
    def update(self, es, **kwargs):
        """change `es.sigma` in place, deprecated?"""
        es.sigma *= self.update2(es, **kwargs)
    def update2(self, es, **kwargs):
        """return sigma change factor
        """
        if not self.is_initialized:
            self.initialize(es)
        return self.sigma * _norm(es.mean) / es.sigma

_CSA_recompute_params_when_masked = True
'''When `True`, recompute cs and ds when the evolution path is masked which
   changes the effective dimension. This will interfere with a possible dynamic
   choice of cs or ds via PPO.'''
_CSA_cs_sqrt = False
_CSA_cs = None  # compute_cs returns this value when not None
_CSA_cs_max = 1/2  # could be 0.3 too (see below)?
_CSA_damps = None
'''Value for CSA damping d_sigma (ds), replaces the mu-dependent default computation
   and is dynamically multiplied by the 'CSA_dampfac' option value'''
_CSA_dampfac_mueff = 2  # was (always) 2
'''Damping for large mueff, the default was 2, however 10 would solve issue #231?'''
_CSA_dampfac_mueff_inner = 3  # smaller is worse on the sectorsphere(44) lam=300
'''Damping factor for mueff > n >> 1, the default was 1'''  # see rerun-issue231 and damps-for-mueff-sweeps
_CSA_dampfac_mueff_attenuation_dimension = 9  # largest single digit
'''Damping attenuation for small dimension such that `_CSA_dampfac_mueff_inner`
   is multiplied by [1/2, 3/4, 7/8,...] when ``dimension ==
   attentuation_dimension * array([1, 2, 3,...])`` and with a smaller factor in
   smaller dimension bounded from below to keep dampfac >= 1.'''
_true_inner = True  # for a quick check; False was rejected
_inner_threshold = 1  # for a quick check, was 1; 2 was rejected
class CMAAdaptSigmaCSA(CMAAdaptSigmaBase):
    """CSA cumulative step-size adaptation AKA path length control.

    As of 2026, CSA is considered as the default step-size control method within
    CMA-ES. It's most notable case of poor performance is with a large neutral
    subspace dimensionality (a large number of "ineffective dimensions"), say
    Neff < N/2.

    Details: only method `hsig` is used from the `CMAAdaptSigmaBase` class.
    """
    def __init__(self, **kwargs):
        """postpone initialization to a method call where dimension and mueff are known
        """
        self.dampdown_fac = csa_dampdown_fac  # immediate setting seems less error prone (more predictable)
        '''Asymmetric damping parameter read from module variable
           `csa_dampdown_fac` (this may change in future)'''
        self.is_initialized = False
    def initialize(self, es, reset=False):
        """set parameters and state variables

        based on dimension, mueff and possibly further options. Attributes are
        not overwritten unless their value is `None` or ``bool(reset) is True``.
        This prevents to overwrite manual settings with late initialization.

        >>> import cma  # test value of damps and reset behavior (minor)
        >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, {'verbose':-9})
        >>> assert 1.479176368423 < es.adapt_sigma.damps < 1.479176368424, es.adapt_sigma.__dict__
        >>> es = cma.CMAEvolutionStrategy(2 * [1], 1, {'verbose':-9})
        >>> assert 1.5 <= es.adapt_sigma.damps < 1.57317317, es.adapt_sigma.__dict__
        >>> es.adapt_sigma.damps = 1.234
        >>> _ = es.adapt_sigma.initialize(es)
        >>> assert es.adapt_sigma.damps == 1.234, es.adapt_sigma.__dict__
        >>> _ = es.adapt_sigma.initialize(es)
        >>> assert es.adapt_sigma.damps == 1.234, es.adapt_sigma.__dict__
        >>> es.adapt_sigma.is_initialized = False
        >>> _ = es.adapt_sigma.initialize(es)
        >>> assert es.adapt_sigma.damps == 1.234, es.adapt_sigma.__dict__
        >>> _ = es.adapt_sigma.initialize(es, reset=True)  # reset everything
        >>> assert 1.5 <= es.adapt_sigma.damps < 1.57317317, es.adapt_sigma.__dict__
        >>> es.adapt_sigma.damps = 1.234
        >>> _ = es.optimize(cma.ff.elli, iterations = 4)
        >>> assert es.adapt_sigma.damps == 1.234, es.adapt_sigma.__dict__

        """
        if self.is_initialized is True and not reset:
            return self
        # by default, do not reset external pre-settings e.g. for cs and damps
        self.initialize_settings(es, reset=reset)
        self.ps = np.zeros(es.N)
        self._ps_updated_iteration = -1
        self.delta = 1
        self.is_initialized = True
        return self
    def initialize_settings(self, es, reset=False):
        """set attributes ``cs, damps`` and others,

        namely ``exponent, disregard_length_setting, max_delta_log_sigma``.

        Attributes are not overwritten when they already exist unless their
        value is `None` or ``bool(reset) is True``.

        Details: We don't need to adjust for a change of cs? A changed cs is
        used in the decay and in the sigma update, hence the latter already
        adjusts for the change in decay. The latter does not adjust the change
        in the entry weight ``sqrt(cs * (2 - cs))`` as both change the same way
        which seems correct too.
        """
        self._es_opts = es.opts  # for the record and possibly later resetting
        # es.opts.normalize_value('CSA_clip_length_value')
        if not reset:
            previous_attributes = dict(self.__dict__)
        self.cs = self.compute_cs(es.N, es.sp.weights.mueff)
        '''`CMAAdaptSigmaBase.hsig` uses this attribute value too'''
        self.damps = self.compute_damps(es.N, es.sp.weights.mueff,
                                        es.sp.popsize, es.sp.lam_mirr)
        # caveat: this would impede sigma increase and hsig, rerun-issue231.ipynb
        if 11 < 3:
            es.opts['CSA_clip_length_value'] = [-es.N / es.popsize / 2, es.N / es.popsize / 2]
            # == 'CSA_clip_length_value': '[-N / popsize / 2, N / popsize / 2]'
            self.damps = (1 + self.cs) * min((2, max((1, np.log(es.popsize / es.N)))))
        self.dampdown_fac = csa_dampdown_fac
        self.max_delta_log_sigma = 1  # in symmetric use (strict lower bound is -cs/damps anyway)

        if es.opts['CSA_disregard_length']:
            if es.opts['CSA_clip_length_value'] is None:
                es.opts['CSA_clip_length_value'] = [0, 0]
            else:
                _warnings.warn("'CSA_disregard_length'={0} was overruled by"
                               " 'CSA_clip_length_value'={1}"
                               .format(es.opts['CSA_disregard_length'],
                                       es.opts['CSA_clip_length_value']))
            # self.damps = es.opts['CSA_dampfac'] * 1  # * (1.1 - 1/(es.N+1)**0.5)
            # if es.opts['verbose'] > 1:
            #     print('CMAAdaptSigmaCSA Parameters: ')
            #     for k, v in self.__dict__.items():
            #         print('  ', k, ':', v)
        if not reset:  # is default
            for name in previous_attributes:
                if previous_attributes[name] is not None:
                    setattr(self, name, previous_attributes[name])
        if self.dampdown_fac != 1 and es.opts['verbose'] > 1:
            _cma_warnings.NeverTestedWarning(
                'CSA damping is asymmetric: dampdown = {0} x dampup'
                .format(self.dampdown_fac))

    def damp_mueff_exponent(self, opts=None):
        """return damping inner exponent to compute damps"""
        if opts is None:
            opts = self._es_opts
        exponent = opts['CSA_damp_mueff_exponent']
        if exponent is None:  # set default
            exponent = 1 if opts['CSA_squared'] else 0.5
        return exponent
    def compute_damps(self, N, mueff, popsize=None, lam_mirr=None, damp_fac=None):
        """return re-computed damps.

        Keyword arguments popsize and lam_mirr are not optional in the first call.

        Depends on the input parameters, on ``self.cs``, on the method
        `damp_mueff_exponent`, and on the module settings
        ``_CSA_dampfac_mueff_inner, _CSA_dampfac_mueff_attenuation_dimension``.
        """
        if _CSA_damps is not None:
            return (self._es_opts['CSA_dampfac'] if damp_fac is None else damp_fac
                    ) * _CSA_damps
        if popsize is not None:
            self._popsize = popsize  # last used popsize
        if lam_mirr is not None:
            self._lam_mirr = lam_mirr
        damp_in, ref_dim = _CSA_dampfac_mueff_inner, _CSA_dampfac_mueff_attenuation_dimension
        damp_in_eff = damp_in if ref_dim <= 1 else max((1,
                            damp_in * (1 - 0.5**(N / ref_dim))))
            # max((1, damp_in * 0.5**(n2 / es.N)))  # -> damp_in for es.N -> infty
            # max((1, damp_in * (1 - n2 / (es.N + n2))))
            # damp_in**(1 - n2 / (es.N + n2))  # see rerun-issue231
        return (1 if damp_fac is None else damp_fac) * (
                0.5
                + min((1, (self._lam_mirr / (0.159 * self._popsize) - 1)**2))**1 / 2
                + _CSA_dampfac_mueff * 
                    damp_in_eff**(1 - _true_inner) * max((0,
                    damp_in_eff**_true_inner *
                    ((mueff-1) / (N+1))**self.damp_mueff_exponent() - _inner_threshold))
                + self.cs
                )
    def compute_cs(self, N, mueff):
        """return a new computation for the decay parameter cs,

        based on the input parameters dimension and mu_w and _CSA_cs_sqrt.

        Details: In Akimoto & Hansen 2020, c_c (not c_sigma) depends on c1 and mueff.

        ``sum((1-c)**i for i = 0,1,...) = 1/c``
        Last contributes half when ``c = 1/2`` ``(1 = 1/c/2)``
        Last two contribute half when ``c = 0.29289322`` ``(1 + (1 - c) = 1/c/2 => c = 1 - sqrt(2) / 2)``
        Alternating contributions: ``1 + (1-c)^2 + ... = alpha * 1 / c``::

           c     alpha  ratio of contributions   bias from order
           1/2   2/3      2:1                      1/3
           0.3   0.588    1.426:1 < 3:2            0.176

        """
        if _CSA_cs is not None:
            return _CSA_cs
        if _CSA_cs_sqrt:
            ## meta_parameters.cs_exponent == 1.0
            b = 1.0 * 0.5
            ## meta_parameters.cs_multiplier == 1.0
            cs = 1.0 * min((_CSA_cs_max, (mueff + 1)**b / (N**b + 2 * mueff**b)))
            return cs
        ## meta_parameters.cs_exponent == 1.0
        b = 1.0
        ## meta_parameters.cs_multiplier == 1.0
        cs = 1.0 * min((_CSA_cs_max, (mueff + 2)**b / (N**b + (mueff + 3)**b)))
        return cs

    def _update_ps(self, es):
        """update path with isotropic delta mean, possibly clipped.

        From input argument `es`, the attributes ``isotropic_mean_shift,
        opts['CSA_clip_length_value'], countiter, sp.weights.mueff,
        path_mask`` are used. opts['CSA_clip_length_value'] can be a single
        value, the upper bound factor, such that::

            max_len = sqrt(N) + opts['CSA_clip_length_value'] * N / (N+2)

        or a list with a lower and an upper factor.
        """
        if not self.is_initialized:
            self.initialize(es)
        if self._ps_updated_iteration == es.countiter:
            return  # don't update twice
        try:  # hasattr(es, name) may trigger the property getter
            idx = getattr(es, 'path_mask')  # a boolean mask
        except AttributeError:
            ("Missing property ``path_mask`` of argument es")
            z = es.isotropic_mean_shift
            Neff = len(z)
            assert Neff == len(self.ps)
        else:
            Neff = np.sum(idx)
            if Neff > 0:  # not needed when Neff == 0
                z = es.isotropic_mean_shift
                if Neff < len(z):
                    z = z[idx]
                assert Neff == len(z), (Neff, len(z), idx)

        ### clip length of z in case
        if Neff > 0 and es.opts.normalize_value('CSA_clip_length_value') is not None:
            vals = es.opts['CSA_clip_length_value']  # the value is normalized (opts.normalize_value was called before)
            N = Neff
            min_len = Mh.chiN(N) + vals[0] * N / (N + 2)
            max_len = Mh.chiN(N) + vals[1] * N / (N + 2)
            act_len = _norm(z)  # assumes that len(z) == Neff
            new_len = Mh.minmax(act_len, min_len, max_len)
            if new_len != act_len:
                z *= new_len / act_len  # assumes that len(z) == Neff
                # z *= (es.N / sum(z**2))**0.5  # ==> sum(z**2) == es.N
                # z *= es.const.chiN / sum(z**2)**0.5
        ### update ps
        if Neff == 0:
            self.ps *= 1 - self.cs
        elif Neff == len(self.ps):  # is not masked, "default"
            self.ps *= 1 - self.cs
            self.ps += _sqrt(self.cs * (2 - self.cs)) * z
        else:
            if _CSA_recompute_params_when_masked:
                self.cs = self.compute_cs(Neff, es.sp.weights.mueff)
                self.damps = self.compute_damps(Neff, es.sp.weights.mueff)
            self.ps *= 1 - self.cs
            self.ps[idx] += _sqrt(self.cs * (2 - self.cs)) * z
        self._ps_updated_iteration = es.countiter

    def update2(self, es, **kwargs):
        """call ``self._update_ps(es)`` and update self.delta.

        Return change factor of self.delta stored in `._last_multiplier`.

        From input `es`, attributes ``opts, countiter, path_for_sigma_update``,
        and sometimes ``_path_for_invariant_update, sp.cc`` are used.
        """
        # We could return self._last_multiplier if self._update_iteration == es.countiter
        self._update_ps(es)
        def masked_path(p):
            # we could use the index from _ps_update but this would
            # in effect reimplement masked_path too
            try:
                return es.masked_path(p)
            except AttributeError:  # should never happen
                ("Argument `es` has no `masked_path(path)`"
                               " method, hence using the identity.")
            return p
        if es.opts.get('CSA_invariant_path'):  # was pc_for_ps = 'pc for ps' in es.opts['vv']
            # was: es.D**-1 * np.dot(es.B.T, es.pc)
            if es.opts['verbose'] > 1 and es.countiter == 1:
                utils.print_message('CSA uses invariant path pc for ps')
            p = masked_path(es._path_for_invariant_update)
            if len(p) < len(self.ps):
                _warnings.warn("Sigma path is masked (len({0}->{1})), but "
                               "'CSA_invariant_path' option does not adjust cs"
                               .format(len(self.ps), len(p)))
            cs = es.sp.cc
        else:
            cs = self.cs
            p = masked_path(self.ps)
        N = len(p)
        if N == 0:  # all variables are masked, nothing to do
            self._last_multiplier = 1
            return 1
        if es.opts['CSA_squared']:
            s = (np.sum(_square(p)) / N - 1) / 2
            # sum(self.ps**2) / es.N has mean 1 and std sqrt(2/N) and is skewed
            # divided by 2 to have the derivative d/dx (x**2 / N - 1) for x**2=N equal to 1
        else:
            s = _norm(p) / Mh.chiN(N) - 1
        s *= cs / self.damps
        try:
            s /= es.opts['CSA_dampfac']
        except Exception as e:
            if _cma_warnings.deliver_warning(self, 'CSA_dampfac multiplication failed'):
                _warnings.warn("s /= es.opts['CSA_dampfac'] failed with {0} ".format(e)
                               + _cma_warnings.deliver_warning.message())
        if s < 0:
            s /= self.dampdown_fac
        s_clipped = Mh.minmax(s, -self.max_delta_log_sigma, self.max_delta_log_sigma)
        # "error" handling
        if s_clipped != s and es.opts['verbose'] > -3:
            _warnings.warn('sigma change np.exp(' + str(s) + ') = ' + str(np.exp(s)) +
                           ' clipped to np.exp(+-' + str(self.max_delta_log_sigma) + ')' +
                           ' \nat iteration ' + str(es.countiter) +
                           ' in CMAAdaptSigmaCSA.update2')
        self._last_multiplier = np.exp(s_clipped)
        self.delta *= self._last_multiplier
        return self._last_multiplier
    def update(self, es, **kwargs):
        """call ``self._update_ps(es)`` and update ``es.sigma``.

        Legacy method replaced by `update2`.
        """
        es.sigma *= self.update2(es, **kwargs)
        # the rest would go to too update2 in case?
        if 11 < 3:
            # derandomized MSR = natural gradient descent using mean(z**2) instead of mu*mean(z)**2
            fit = kwargs['fit']  # == es.fit
            slengths = np.array([sum(z**2) for z in es.arz[fit.idx[:es.sp.weights.mu]]])
            # print lengths[0::int(es.sp.weights.mu/5)]
            es.sigma *= np.exp(np.dot(es.sp.weights, slengths / es.N - 1))**(2 / (es.N + 1))
        if 11 < 3:
            es.more_to_write.append(10**((sum(self.ps**2) / es.N / 2 - 1 / 2 if es.opts['CSA_squared'] else _norm(self.ps) / es.const.chiN - 1)))
            es.more_to_write.append(10**(-3.5 + sum(self.ps**2) / es.N / 2 - _norm(self.ps) / es.const.chiN))
            # es.more_to_write.append(10**(-3 + sum(es.arz[es.fit.idx[0]]**2) / es.N))

class CMAAdaptSigmaMedianImprovement(CMAAdaptSigmaBase):
    """Compares median fitness to the 27%tile fitness of the
    previous iteration, see Ait ElHara et al, GECCO 2013.

    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(3 * [1], 1,
    ... {'AdaptSigma':cma.sigma_adaptation.CMAAdaptSigmaMedianImprovement,
    ...  'verbose': -9})
    >>> assert es.optimize(cma.ff.elli).result[1] < 1e-9
    >>> assert es.result[2] < 2000

    """
    def __init__(self, **kwargs):
        CMAAdaptSigmaBase.__init__(self)  # base class provides method hsig()
        # super(CMAAdaptSigmaMedianImprovement, self).__init__()
    def initialize(self, es):
        """late initialization using attributes ``N`` and ``popsize``"""
        r = es.sp.weights.mueff / es.popsize
        self.index_to_compare = 0.5 * (r**0.5 + 2.0 * (1 - r**0.5) / np.log(es.N + 9)**2) * (es.popsize)  # TODO
        self.index_to_compare = 0.30 * es.popsize  # TODO
        self.damp = 2 - 2 / es.N  # sign-rule: 2
        self.c = 0.3  # sign-rule needs <= 0.3
        self.s = 0  # averaged statistics, usually between -1 and +1
        return self
    def update(self, es, **kwargs):
        if es.countiter < 2:
            self.initialize(es)
            self.fit = es.fit.fit
        else:
            ft1 = self.fit[int(self.index_to_compare)]
            ft2 = self.fit[int(np.ceil(self.index_to_compare))]
            ftt1 = es.fit.fit[(es.popsize - 1) // 2]
            # ftt2 = es.fit.fit[int(np.ceil((es.popsize - 1) / 2))]
            pt2 = self.index_to_compare - int(self.index_to_compare)
            # ptt2 = (es.popsize - 1) / 2 - (es.popsize - 1) // 2  # not in use
            s = 0
            if 1 < 3:
                s += pt2 * sum(es.fit.fit <= self.fit[int(np.ceil(self.index_to_compare))])
                s += (1 - pt2) * sum(es.fit.fit < self.fit[int(self.index_to_compare)])
                s -= es.popsize / 2.
                s *= 2. / es.popsize  # the range was popsize, is 2
            elif 11 < 3:  # compare ft with median of ftt
                s += self.index_to_compare - sum(self.fit <= es.fit.fit[es.popsize // 2])
                s *= 2 / es.popsize  # the range was popsize, is 2
            else:  # compare ftt j-index of ft
                s += (1 - pt2) * np.sign(ft1 - ftt1)
                s += pt2 * np.sign(ft2 - ftt1)
            self.s = (1 - self.c) * self.s + self.c * s
            es.sigma *= np.exp(self.s / self.damp)
        # es.more_to_write.append(10**(self.s))

        #es.more_to_write.append(10**((2 / es.popsize) * (sum(es.fit.fit < self.fit[int(self.index_to_compare)]) - (es.popsize + 1) / 2)))
        # # es.more_to_write.append(10**(self.index_to_compare - sum(self.fit <= es.fit.fit[es.popsize // 2])))
        # # es.more_to_write.append(10**(np.sign(self.fit[int(self.index_to_compare)] - es.fit.fit[es.popsize // 2])))
        if 11 < 3:
            import scipy.stats as _stats
            zkendall = _stats.kendalltau(list(es.fit.fit) + list(self.fit),
                                         len(es.fit.fit) * [0] + len(self.fit) * [1])[0]
            es.more_to_write.append(10**zkendall)
        self.fit = es.fit.fit

def tpa_abs_z_with_offset0(z_abs, params):
    """return `z_abs` unmodified"""
    return z_abs
def tpa_abs_z_with_offset1(z_abs, params):
    """return ``z_abs + z_min``"""
    return 0 if z_abs == 0 else params.z_min + z_abs
def tpa_abs_z_with_offset2(z_abs, params):
    """return absolute z-value `z_abs` lower bounded by `z_min`"""
    return 0 if z_abs == 0 else max((params.z_min, z_abs))
tpa_abs_z_with_offset = tpa_abs_z_with_offset0
'''modify |z| with an offset to boost small changes, needs to be assessed'''
class _TPAParameters(object):
    """Parameters for `CMAAdaptSigmaTPA`,

    namely ``damp, c, dampdown_fac, z_exponent, z_min`` will be assigned in
    `CMAAdaptSigmaTPA.initialize`.
    """
class CMAAdaptSigmaTPA(CMAAdaptSigmaBase):
    """two point adaptation for step-size sigma.

    Relies on a specific sampling of the first two offspring, whose
    objective function value ranks are used to decide on the step-size
    change, see `update` for the specifics.

    Example
    =======

    >>> import cma
    >>> cma.CMAOptions('adapt').pprint()  # doctest: +ELLIPSIS
     AdaptSigma='True...
    >>> es = cma.CMAEvolutionStrategy(10 * [0.2], 0.1,
    ...     {'AdaptSigma': cma.sigma_adaptation.CMAAdaptSigmaTPA,
    ...      'ftarget': 1e-8})  # doctest: +ELLIPSIS
    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 10 (seed=...
    >>> es.optimize(cma.ff.rosen)  # doctest: +ELLIPSIS
    Iter...
    >>> assert 'ftarget' in es.stop()
    >>> assert es.result[1] <= 1e-8  # should coincide with the above
    >>> assert es.result[2] < 6500  # typically < 5500

    References: loosely based on Hansen 2008, CMA-ES with Two-Point
    Step-Size Adaptation, more tightly reflecting Hansen et al. 2014,
    How to Assess Step-Size Adaptation Mechanisms in Randomized Search
    and Akimoto & Hansen 2020, Diagonal Acceleration for Covariance...

    TODO: collect data for ``._last_z`` and/or ``.s`` distributions, namely on
    the stationary sphere with ``sigma in [sigma_opt, 2 * sigma_opt]`` or as a
    function of the convergence rate and depending on popsize: is ``|s|``
    decreasing with increasing popsize and how? Can we compute hsig based on s
    instead ps?
"""
    def __init__(self, dimension=None, opts=None, **kwargs):
        """`popsize` is a valid `kwargs`"""
        super(CMAAdaptSigmaTPA, self).__init__() # base class provides method hsig()
        # CMAAdaptSigmaBase.__init__(self)
        self.initialized = False
        self.dimension = dimension
        self._es_opts = opts
        self._popsize = kwargs.get('popsize', None)
        self.sp = _TPAParameters()
        '''parameter settings'''
        self.sp.dampdown_fac = tpa_dampdown_fac  # seems less error prone (more predictable) than lazy init
        '''Asymmetric damping parameter read from module variable
           `tpa_dampdown_fac` (this may change in future)'''
        self.s = 0
        '''the state/summation variable'''
        self.delta = 1
        '''all sigma changes multiplied'''
    def initialize(self, N=None, opts=None, popsize=None, reset=False, **kwargs):
        """late initialization based on `CMAEvolutionStrategy`.

        Argument `N` is either the dimension (for backward compatibility) or a
        `CMAEvolutionStrategy` instance. `opts` and `popsize` are ignored in the
        latter case.

        Argument `opts` is equivalent with ``N.opts``, used for
        `'TPA_dampfac'`, verbosity behavior and hacking (very versatile).

        Unless ``bool(reset) is True``, `initialize` does not overwrite
        parameters or state variables unless their value is `None` or they are
        in the `_attributes_to_not_recover_on_init` list.

        The following mainly tests the (new) reset behavior:

        >>> import cma  # test sp.damp value and reset behavior (minor)
        >>> es = cma.CMAEvolutionStrategy(2 * [1], 1, {'verbose':-9,
        ...               'AdaptSigma': cma.sigma_adaptation.CMAAdaptSigmaTPA})
        >>> assert 4.85888 < es.adapt_sigma.sp.damp < 4.85889, es.adapt_sigma.sp.__dict__
        >>> es.adapt_sigma.sp.damp = 1.234
        >>> _ = es.adapt_sigma.initialize(es)
        >>> assert es.adapt_sigma.sp.damp == 1.234, es.adapt_sigma.sp.__dict__
        >>> _ = es.adapt_sigma.initialize(es)
        >>> assert es.adapt_sigma.sp.damp == 1.234, es.adapt_sigma.sp.__dict__
        >>> es.adapt_sigma.is_initialized = False
        >>> _ = es.adapt_sigma.initialize(es)
        >>> assert es.adapt_sigma.sp.damp == 1.234, es.adapt_sigma.sp.__dict__
        >>> _ = es.adapt_sigma.initialize(es, reset=True)  # reset everything
        >>> assert 4.85888 < es.adapt_sigma.sp.damp < 4.85889, es.adapt_sigma.sp.__dict__
        >>> es.adapt_sigma.sp.damp = 1.234
        >>> _ = es.optimize(cma.ff.elli, iterations = 4)
        >>> assert es.adapt_sigma.sp.damp == 1.234, es.adapt_sigma.sp.__dict__

        """
        if self.initialized is True and not reset:
            return self

        if hasattr(N, 'N'):  # kinda type test
            self.initialize_settings(N, reset)
        else:
            if N is not None:
                self.dimension = N
            if opts is not None:  # keep previous when None
                self._es_opts = opts
            if popsize is not None:
                self._popsize = popsize
            self.initialize_settings(None, reset)
        if reset:  # cp-paste from __init__
            self.s = 0
            self.delta = 1
        self.initialized = True
        return self
    def initialize_settings(self, es=None, reset=False):
        """set parameters in ``.sp`` based on an `CMAEvolutionStrategy` instance

        or on the previously stored ``.dimension, ._es_opts, ._popsize``. These
        attributes could be modified to compute a respective setting. Any
        existing setting is overwritten only if ``bool(reset) is True`` or the
        respective parameter value in ``.sp`` is `None`.
        """
        ### Prepare
        _N = self.dimension
        if es is not None:
            self._es_opts = es.opts
            self._popsize = es.sp.popsize
            self.dimension = es.N
        if not reset:  # recover already set attributes in the end
            previous_sp = dict(self.sp.__dict__)
            if self.dimension != _N:
                _warnings.warn("Previous dimension = {0} != {1} = new dimension."
                               "\nThis may lead to an inconsistent setting of"
                               " TPA parameters {2}."
                               .format(_N, self.dimension, self.sp.__dict__))
        ### Go
        N, opts, popsize = self.dimension, self._es_opts, self._popsize
        try:
            # Suggested for self.sp.damp:
            #   N**0.5          # (1)
            #   4 - 3.6/N**0.5  # (2) should become new default!?
            #   N**0.25
            #   0.7 + np.log(N)  # between 2 and 9 very close to N**1/2, for N=7 equal to (1) and (2)
            self.sp.damp = 0.7 + 2 * np.log(N)
            self.sp.damp += 2 * np.log(max((1, popsize - N)))  # fix issue 231
        except Exception as e:
            _warnings.warn("Setting TPA damping failed with exception {0} (self={1}, self.sp={2})"
                           .format(e, self.__dict__, self.sp.__dict__))
            self.sp.damp = 4  # or 1 + np.log(10)
            # self.initialized = 1/2
        try:
            self.sp.damp = opts['vv']['TPA_damp']
            utils.print_message('TPA damping set from option vv={0} to {1}'
                                .format(opts['vv'], self.sp.damp))
        except (KeyError, TypeError):
            pass

        # self.sp.dampup = 0.5**0.0 * 1.0 * self.sp.damp  # 0.5 fails to converge on the Rastrigin function
        # self.sp.dampdown = 2.0**0.0 * self.sp.damp
        self.sp.dampdown_fac = tpa_dampdown_fac
        self.sp.c = 0.3  # rank difference is asymmetric(?) and therefore the switch from increase to decrease takes too long?
        '''with the default averaging coefficient, the first two entries dominate all others:
           c = 1/2 <==> 1/c * 1/2 == 1 meaning the first entry == 1 takes half of the weight
           c = 0.29289 = 1 - sqrt(1/2) <==> 1/c * 1/2 == 1 + 1-c meaning the first two
           entries take half of the weight, where "the full" weight is 1/c = sum_{i=0}^infty (1-c)^i'''
        self.sp.z_exponent = 0.5  # sign(z) * abs(z)**z_exponent, 0.5 seems better with larger popsize, 1 was default
        '''exponent < 1 enlarges small z-values, originally ``z_exponent == 1``'''
        self.sp.z_min = 0.1
        '''used by `tpa_z_abs_with_offset`, set by default the minimal |z| such
           that ``exp(|z/damp|) >= exp(z_min/damp)`` if z != 0. By design, we
           have ``z_min >= 1 / (lambda-1)``.'''
        # self.sp.sigma_fac = 1.0  # (obsolete) 0.5 feels better, but no evidence whether it is
        # self.sp.relative_to_delta_mean = True  # (obsolete)
        if not reset:  # recover original values in case they were already set
            for name in previous_sp:
                if previous_sp[name] is not None:
                    setattr(self.sp, name, previous_sp[name])
        if self.sp.dampdown_fac != 1 and (opts is None or opts.get('verbose', 0) > 1):
            _cma_warnings.NeverTestedWarning('TPA damping is asymmetric: dampdown = {0} x dampup'
                                             .format(self.sp.dampdown_fac))
        return self
    def update2(self, es, function_values, **kwargs):
        """update state variables and return the step-size multiplier.

        The first and second value in ``function_values`` must reflect two
        mirrored solutions sampled, respectively, in direction and in opposite
        direction of the previous mean shift.
        """
        # On the linear function, the two mirrored samples lead
        # to a sharp increase of the condition of the covariance matrix,
        # unless we have negative weights (which we have now by default).
        # Otherwise they should not be used to update the covariance
        # matrix, if the step-size inreases quickly.
        if self.initialized is not True:  # try again
            self.initialize(es)
        if self.initialized is not True:
            _warnings.warn("dimension not known, damping set to {0}".format(self.sp.damp))
            self.initialized = True
        if 1 < 3:
            f_vals = np.asarray(function_values)
            z = np.sum(f_vals < f_vals[1]) - np.sum(f_vals < f_vals[0])
            z /= len(f_vals) - 1  # z in [-1, 1]
        elif 1 < 3:
            # use the ranking difference of the mirrors for adaptation
            # damp = 5 should be fine
            z = np.nonzero(es.fit.idx == 1)[0][0] - np.nonzero(es.fit.idx == 0)[0][0]
            z /= es.popsize - 1  # z in [-1, 1]
        self.s *= 1 - self.sp.c
        self.s += self.sp.c * np.sign(z) * tpa_abs_z_with_offset(
                               np.abs(z)**self.sp.z_exponent, self.sp)
        self._last_z = z
        try:
            damp_fac = es.opts['TPA_dampfac']
        except (TypeError, KeyError, AttributeError) as e:
            if _cma_warnings.deliver_warning(self, 'TPA_dampfac access failed'):
                _warnings.warn("damp_fac = es.opts['TPA_dampfac'] failed with {0} ".format(e)
                               + _cma_warnings.deliver_warning.message())
            damp_fac = 1
        self._last_multiplier = np.exp(self.s / self.sp.damp / damp_fac
                                       / (self.sp.dampdown_fac if self.s < 0 else 1))
        self.delta *= self._last_multiplier
        #es.more_to_write.extend([10**z, 10**self.s])
        return self._last_multiplier
    def update(self, es, function_values, **kwargs):
        """update ``es.sigma *= self.update2(...)``.

        The first and second value in ``function_values`` must reflect two
        mirrored solutions.

        Legacy method replaced by `update2`.
        """
        es.sigma *= self.update2(es, function_values, **kwargs)

    def check_consistency(self, es):
        assert isinstance(es.adapt_sigma, CMAAdaptSigmaTPA)
        if es.countiter > 3:
            j = np.random.randint(0, es.N)
            dm = es.mean_after_tell[j] - es.mean_old[j]
            dx0 = es.pop[0][j] - es.mean[j]
            dx1 = es.pop[1][j] - es.mean[j]
            for i in np.random.randint(0, es.N, 1):
                if i == j:
                    i -= 1 if i > 0 else -1
                if dx0 * dx1 * (es.pop[0][i] - es.mean[i]) * (
                    es.pop[1][i] - es.mean[i]):
                    dmi_div_dx0i = (es.mean_after_tell[i] - es.mean_old[i]) \
                                    / (es.pop[0][i] - es.mean[i])
                    dmi_div_dx1i = (es.mean_after_tell[i] - es.mean_old[i]) \
                                        / (es.pop[1][i] - es.mean[i])
                    expected_precision = 1e-11 * max((1, np.abs(es.mean[i]) / es.stds[i],
                                                         np.abs(es.mean[j]) / es.stds[j]))
                    # was: expected_precision = 1e-4
                    if not Mh.equals_approximately(
                            dmi_div_dx0i, dm / dx0, expected_precision) or \
                            not Mh.equals_approximately(
                                    dmi_div_dx1i, dm / dx1, expected_precision):
                        m = utils.format_warning(
                            'TPA: apparent inconsistency with mirrored'
                            ' samples, where dmi_div_dx0i, dm/dx0=%f, %f'
                            ' and dmi_div_dx1i, dm/dx1=%f, %f'
                            ' \n i=%d expected precision=%f mean[i]=%f' % (
                                dmi_div_dx0i, dm/dx0, dmi_div_dx1i, dm/dx1,
                                i, expected_precision, es.mean[i]),
                            'check_consistency',
                            'CMAAdaptSigmaTPA', es.countiter); m and _warnings.warn(m)
                else:
                    m = utils.format_warning('zero delta encountered in TPA which' +
                                        ' \nshould be very rare and might be a bug' +
                                        ' (sigma=%f)' % es.sigma,
                                        'check_consistency', 'CMAAdaptSigmaTPA',
                                        es.countiter); m and _warnings.warn(m)
