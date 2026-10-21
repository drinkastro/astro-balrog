import numpy as np
import matplotlib.pyplot as plt
import emcee
from measure_BR import measure_BR
from _init_ import path_to_models

def _interp(y0, y1, p0, p1, p2, plot=False):
    y = y0 + (p0 - p1) * (y1 - y0) / (p2 - p1)
    if plot:
        plt.plot(y0, 'b',
                 y1, 'b',
                 y, 'r')
    return y


def _grid_params_smiles():
    """
    Return the parameters of the sMILES models as numpy arrays.
    """
    ages = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15,
                     0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7,
                     0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5,
                     2.75, 3., 3.25, 3.5, 3.75, 4., 4.5, 5., 5.5,
                     6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.,
                     10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.])
    mets = np.array([-1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25, 0.06,
                     0.15, 0.26])
    afs = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
    slopes = np.array([0.3, 0.5, 0.8, 1.3, 1.5, 1.8, 2., 2.3, 2.5, 2.8, 3., 3.3, 3.5])

    return ages, mets, afs, slopes


def _interp_mod(templates, age, met, afe, fixed_imf_slope=1.3):
    """
    Return a model of given age, met, and [a/Fe], at a fixed imf.
    This routine is written to deal with the sMILES models,
    but it is easily adaptable to any library, once the corresponding params are provided
    through a customized _grid_params routine.

    :param templates: all smiles templates
    :param age: age to interpolate
    :param met: metallicity to interpolate
    :param afe: [a/Fe] to interpolate
    :param fixed_imf_slope: imf slope of the bimodal parametrization. Must be a value included in the grid params, not interpolating here. Default: Milky-Way-like.
    :return: model flux interpolated at age, met, afe, and imf
    """

    ages, metals, afes, slopes = _grid_params_smiles()
    wa = np.array([np.where(ages < age)[0][-1], np.where(ages >= age)[0][0]])

    wm = np.array([np.where(metals < met)[0][-1], np.where(metals >= met)[0][0]])
    waf = np.array([np.where(afes < afe)[0][-1],
                    np.where(afes >= afe)[0][0]])

    a1, a2 = ages[wa][0], ages[wa][1]
    m1, m2 = metals[wm][0], metals[wm][1]
    af1, af2 = afes[waf][0], afes[waf][1]

    ws = slopes == fixed_imf_slope

    # interpolate met
    ym1 = _interp(templates[:, wa[0], wm[0], waf[0], ws].ravel(),
                 templates[:, wa[0], wm[1], waf[0], ws].ravel(), met, m1, m2)  # a1af1s1
    ym2 = _interp(templates[:, wa[1], wm[0], waf[0], ws].ravel(),
                 templates[:, wa[1], wm[1], waf[0], ws].ravel(), met, m1, m2)  # a2af1s1
    ym3 = _interp(templates[:, wa[0], wm[0], waf[1], ws].ravel(),
                 templates[:, wa[0], wm[1], waf[1], ws].ravel(), met, m1, m2)  # a1af2s1
    ym4 = _interp(templates[:, wa[1], wm[0], waf[1], ws].ravel(),
                 templates[:, wa[1], wm[1], waf[1], ws].ravel(), met, m1, m2)  # a2af2s1

    # interpolate age
    ya1 = _interp(ym1, ym2, age, a1, a2)  # af1s1
    ya2 = _interp(ym3, ym4, age, a1, a2)  # af2s1

    # interpolate [a/Fe]
    y = _interp(ya1, ya2, afe, af1, af2)

    return y

def _calchi2(lam, templates, age, met, afe,  br_obs, br_err,
                blist, sumn):
    temp = _interp_mod(templates, age, met, afe).ravel()
    br_mod = measure_BR(lam, temp, blist=blist, sumn=sumn, err=None)[0]
    chi2 = (br_obs - br_mod) ** 2 / br_err ** 2
    return chi2


def _log_prior(theta, age_bf, age_err, afe_bf=None, afe_err=None, prior='uniform'):
        age, met, afe = theta
        amin, amax = age_bf - age_err, age_bf + age_err
        metmin, metmax = -1.79, 0.26 #Only considering the safe metallicity ranges (see Vazdekis+15, Knowles+23)

        if afe_bf is not None:
            afemin, afemax = afe_bf - afe_err, afe_bf+afe_err #Allowing to constrain [a/Fe], if an independent measurement is available
        else:
            afemin, afemax = -0.2, 0.6 #min and max values available from smiles models

        #Avoid numerical issues if the best-fit age is too young and/or errors are too large.
        #I set amin=0.1 because at <100Myr models are unsafe (see Vazdekis+15)
        #and because the metallicity from the Balmer Ratio is too degenerate (see Bevacqua+26).
        if age_bf - age_err < 0.1:
            amin = 0.1

        if amin <= age <= amax and metmin <= met <= metmax and afemin <= afe <= afemax:
            return 0.0

        return -np.inf

def _log_likelihood(theta, lam_temp, templates, br_obs, br_err, blist, sumn):
        age, met, afe = theta
        chi2 = _calchi2(lam_temp, templates, age, met, afe, br_obs, br_err, blist, sumn)
        return -0.5 * chi2

def _log_prob(theta, lam_temp, templates, age_bf, age_err, br_obs, br_err, blist, sumn, afe_bf=None, afe_err=None):
        lp = _log_prior(theta, age_bf, age_err, afe_bf=afe_bf, afe_err=afe_err)
        if not np.isfinite(lp):
            return -np.inf
        return lp + _log_likelihood(theta, lam_temp, templates, br_obs, br_err, blist, sumn)


def _ppxf_varsmooth(x, y, sig_x, xout=None, oversample=1):
    """
    ***********************************************************************
    COPY-AND-PASTED, WITH PERMISSION, FROM pPXF.
    THANKS TO MICHELE CAPPELLARI FOR ALLOWING ME TO INCLUDE IT IN THIS CODE.
    ***********************************************************************
    Fast and accurate convolution with a Gaussian of variable width.

    This function performs an accurate Fourier convolution of a vector, or the
    columns of an array, with a Gaussian kernel that has a varying or constant
    standard deviation (sigma) per pixel. The convolution is done using fast
    Fourier transform (FFT) and the analytic expression of the Fourier
    transform of the Gaussian function, like in the pPXF method. This allows
    for an accurate convolution even when the Gaussian is severely
    undersampled.

    This function is recommended over standard convolution even when dealing
    with a constant Gaussian width, due to its more accurate handling of
    undersampling issues.

    This function implements Algorithm 1 in `Cappellari (2023)
    <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_

    Parameters
    ----------
    x : array_like
        Coordinate of every pixel in `y`.
    y : array_like
        Input vector or array of column-spectra.
    sig_x : float or array_like
        Gaussian sigma of every pixel in units of `x`.
        If sigma is constant, `sig_x` can be a scalar.
        In this case, `x` must be uniformly sampled.
    oversample : float, optional
        Oversampling factor before convolution (default: 1).
    xout : array_like, optional
        Output `x` coordinate used to compute the convolved `y`.

    Returns
    -------
    yout : array_like
        Convolved vector or columns of the array `y`.

    """
    assert len(x) == len(y), "`x` and `y` must have the same length"

    def ppxf_interp(xout, xin, yin):
        """Applies `numpy.interp` to the last dimension of `yin`
        COPY-AND-PASTED FROM pPXF. NEEDED FOR ppxf_varsmooth."""

        yout = [np.interp(xout, xin, y) for y in yin.reshape(-1, xin.size)]

        return np.reshape(yout, (*yin.shape[:-1], -1))

    if np.isscalar(sig_x):
        dx = np.diff(x)
        assert np.all(np.isclose(
            dx[0], dx)), "`x` must be uniformly spaced, when `sig_x` is a scalar"
        n = len(x)
        sig_max = sig_x * (n - 1) / (x[-1] - x[0])
        y_new = y.T
    else:
        assert len(x) == len(sig_x), "`x` and `sig_x` must have the same length"

        # Stretches spectrum to have equal sigma in the new coordinate
        sig = sig_x / np.gradient(x)
        sig = sig.clip(0.1)  # Clip to >=0.1 pixels
        sig_max = np.max(sig) * oversample
        xs = np.cumsum(sig_max / sig)
        n = int(np.ceil(xs[-1] - xs[0]))
        x_new = np.linspace(xs[0], xs[-1], n)
        y_new = ppxf_interp(x_new, xs, y.T)

    # Convolve spectrum with a Gaussian using analytic FT like pPXF
    npad = 2 ** int(np.ceil(np.log2(n)))
    ft = np.fft.rfft(y_new, npad)
    w = np.linspace(0, np.pi * sig_max, ft.shape[-1])
    ft_gau = np.exp(-0.5 * w ** 2)
    yout = np.fft.irfft(ft * ft_gau, npad)[..., :n]

    if not np.isscalar(sig_x):
        if xout is not None:
            xs = np.interp(xout, x, xs)  # xs is 1-dim

        yout = ppxf_interp(xs, x_new, yout)

    return yout.T


def fit_BR(lam, flux, err, age_bf, age_err, sigma_gal=None,
               blist = ['Hgamma_F', 'Hdelta_F', 'Hzeta', 'Htheta', 'Hiota', 'Hbeta', 'Heta'],sumn = 5,
               afe_best=None, afe_err=None, nwalkers=30, nsteps=10000, discard=5000, thin=15):

    """
    *** Main routine ***
    Fit the observed Balmer Ratio (BR) with model predictions to obtain the metallicity.
    The best-fit age is a mandatory parameter because the BR is degenerate with it,
    so an independent estimate of the age is needed (see Bevacqua et al. 2026).

    Mandatory parameters
    :param lam: (array) wavelengths in Angstrom
    :param flux: (array) flux in arbitary units
    :param err: (array) error on flux in arbitary units (but same units as flux)
    :param age_bf: (float) best-fitting age
    :param age_err: (float) uncertainty on the best-fitting age. The fit will be constrained within [afe_best-afe_err, afe_best+afe_err].

    Optional parameters:
    :param sigma_gal: (float) stellar velocity dispersion of the galaxy. If None, the spectral indices of the models are calculated at sigma_instrumental
    :param blist: (list) Balmer spectral indices to be used to compute the BR.
    :param sumn: (int) number of the sumn-th balmer indices to be summed up. The ratio is computed as sum(blist[:sumn]) / sum(blist[sumn:)),
    :param afe_best: (float) best-fitting [a/Fe], if available from an independent measurement, or to constrain the fit to a fixed value.
    :param afe_err:  (float) error on the best-fitting [a/Fe]. The fit will be constrained within [afe_best-afe_err, afe_best+afe_err].
    :param nwalkers: (int) number of walkers for the emcee sampling
    :param nsteps: (int) number of steps for the emcee sampling
    :param discard: (int) number of points to discard from the posterior
    :param thin: (int) for emcee
    :return: posterior samples from emcee
    """


    br_obs, br_err = measure_BR(lam, flux, err=err, blist=blist, sumn=sumn)
    print("Fitting Balmer Ratio:")
    print("(","+".join(blist[:sumn]), ") / (", "+".join(blist[sumn:]), ") = ", round(br_obs, 2), "+/-", round(br_err, 2))

    l = np.load(path_to_models)
    lam_temp, templates, ages, metals = l["lam"], l["templates"], l["ages"], l["metals"]

    tshape = templates.shape[1:]
    if sigma_gal is not None:
        #Smooth templates to match the stellar velocity dispersion of the galaxy
        for i in range(tshape[0]):
            for j in range(tshape[1]):
                for k in range(tshape[2]):
                    for l in range(tshape[3]):
                        c = 299792.458
                        fwhm_tem = 2.51 #FWHM of the sMILES models at the wavelength of the balmer line
                        fwhm_g = 2.355*sigma_gal*lam/c
                        fwhm_gal = np.interp(lam_temp, lam, fwhm_g)
                        fwhm_diff2 = (fwhm_gal ** 2 - fwhm_tem ** 2).clip(0)
                        sigma = np.sqrt(fwhm_diff2) / np.sqrt(4 * np.log(4))
                        templates[:, i, j, k, l] = _ppxf_varsmooth(lam_temp, templates[:, i, j, k, l], sigma)

    ndim = 3

    guess = np.stack(
        [np.random.uniform(age_bf - age_err, age_bf + age_err, nwalkers),  #Initial guess for age
         np.random.uniform(-1.79, 0.26, nwalkers),                         #Initial guess for metallicity
         np.random.uniform(-0.2, 0.6, nwalkers)]).T                        #Initial guess for [a/Fe]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_prob,
                                    args=(lam_temp, templates, age_bf, age_err, br_obs, br_err, blist,
                                          sumn, afe_best, afe_err))

    # Run MCMC
    sampler.run_mcmc(guess, nsteps, progress=True)
    samples = sampler.get_chain(discard=discard, thin=thin,flat=True)

    return samples
