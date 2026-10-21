import numpy as np
import os
from . import path_to_balmers



def _specidx_definitions():
    """Import balmer spectral index definitions from the table balmer.dat"""

    defs = {}
    for line in open(path_to_balmers):
        p = line.split("#")[0].split()
        if p:
            fb, fr, lb, lr, rb, rr = map(float, p[1:7])
            defs[p[0]] = (lb, lr, fb, fr, rb, rr)
    return defs

def _ew(wave, flux, d):
    """Equivalent width (A): linear pseudo-continuum through side-band means."""
    b0, b1, i0, i1, r0, r1 = d
    if wave[0] > min(d) or wave[-1] < max(d):
        return np.nan
    e = np.r_[wave[0] - (wave[1] - wave[0]) / 2, (wave[1:] + wave[:-1]) / 2,
              wave[-1] + (wave[-1] - wave[-2]) / 2]          # pixel edges
    lo, hi = e[:-1], e[1:]
    w = lambda a, b: np.clip(hi, a, b) - np.clip(lo, a, b)  # overlap width per pixel
    mean = lambda a, b: (w(a, b) * flux).sum() / w(a, b).sum()
    fb, fr = mean(b0, b1), mean(r0, r1)
    cb, cr = (b0 + b1) / 2, (r0 + r1) / 2
    wi = w(i0, i1)
    c = fb + (fr - fb) / (cr - cb) * ((np.clip(hi, i0, i1) + np.clip(lo, i0, i1)) / 2 - cb)
    m = wi > 0
    return float(np.sum(wi[m] * (1 - flux[m] / c[m])))


def _compute_ratio(pars, k, errs=None):
    """
    Compute the Balmer ratio and its error.
    This function sums pars[:k]/pars[k:] --> (H_0 + ... + H_k-1)/(H_k + ... + H_n), where n = len(pars).
    The error is calculated from standard error propagation as (num/den) * sqrt((sigma_num/num)**2 + (sigma_den/num)**2)
    :param pars: balmer lines
    :param k: index to split the balmers to be summed
    :param errs: errors on balmer lines (optional)
    :return: balmer ratio and propagated error (if errs is not None)
    """
    N = sum(pars[:k])
    D = sum(pars[k:])
    ratio = N/D
    if errs is None:
        err_ratio = 0
    else:
        sigma_N = np.sqrt(sum(errs[:k]**2)) #Sum from 0 to k-1 (sigma_i **2)
        sigma_D = np.sqrt(sum(errs[k:]**2)) #Sum from k to n (sigma_i **2)
        err_ratio = ratio * np.sqrt(np.square(sigma_N/N) + np.square(sigma_D/D))

    return ratio, err_ratio

def _measure_indices(wave, flux, spec_idx_list, err=None, n_mc=500):
    """
    Measure the strengths of the Balmer lines.

    :param wave: array containing the wavelengths in Ang
    :param flux: array containing the flux of the galaxy in arbitrary units
    :param spec_idx_list: list of balmer lines to be measured
    :param err: array containing the error on the flux in arbitrary units. If none, no errors are computed.
    :param n_mc: number of monte carlo simulation for the flux to get meaningful errors
    :return: dictionary containing the strengths (.strenghts) and uncertainties (.errors) on the measured spectral indices
    """

    from types import SimpleNamespace
    wave, flux = np.asarray(wave, float), np.asarray(flux, float)
    ok = np.isfinite(flux) #remove bad pixels

    if err is not None:
        err =  np.asarray(err, float)
        ok = ok & np.isfinite(err) & (err > 0)  # remove bad pixels
        err = err[ok]

    wave, flux = wave[ok], flux[ok]


    edges = np.r_[wave[0] - (wave[1] - wave[0]) / 2, (wave[1:] + wave[:-1]) / 2,
                  wave[-1] + (wave[-1] - wave[-2]) / 2]
    defs = _specidx_definitions()

    if err is not None:
        rng = np.random.default_rng(0)
        err = err[ok]
        noisy = [flux + rng.normal(0, err) for _ in range(n_mc)] #realize n_mc simulation of the spectrum to compute errors

    strengths, errors = [], []
    for name in spec_idx_list:
        d = defs[name]
        if wave[0] > min(d) or wave[-1] < max(d):
            strengths.append(np.nan);
            errors.append(np.nan)
            print("Warning: the spectrum does not cover", name)
            continue
        strengths.append(_ew(wave, flux, d))
        if err is not None:
            errors.append(float(np.std([_ew(wave, f, d) for f in noisy], ddof=1)))
        else: errors.append(-99.)

    return SimpleNamespace(idx_name=list(spec_idx_list), strengths=strengths, errors=errors)

def measure_BR(wave, flux, err = None, blist=['Hgamma_F', 'Hdelta_F', 'Hiota', 'Hzeta','Htheta','Hbeta', 'Heta'], sumn=5):
    """
    Measure the Balmer ratio and its error.
    The ratio is defined based on the input list of Balmer lines.
    The ratio is computed as sum(blist[:sumn]) / sum(blist[sumn:)),
    namely (H_0 + ... + H_[sumn-1]) / (H_[sumn] + ... + H_[len(blist)]).
    In this regard, the order of the blist matters.

    :param wave: array containing the wavelengths in Ang
    :param flux: array containing the flux of the galaxy in arbitrary units
    :param err: array containing the errors on the flux in arbitrary units (but same as flux). If none, no errors are computed.
    :param blist: list of Balmer lines to compute the ratio.
    :param sumn: number of the index to split blist to get numerator and denominator of the Balmer ratio.
    :return: measured ratio and error. If err is None, return array full of -99.
    """

    indices = _measure_indices(wave, flux, err=err, spec_idx_list=blist)
    strengths, errors = indices.strengths, indices.errors
    strengths, errors = np.asarray(strengths), np.asarray(errors)
    if err is not None:
        ratio, err_ratio = _compute_ratio(strengths, sumn, errors)
    else:
        ratio, err_ratio = _compute_ratio(strengths, sumn, err)

    return ratio, err_ratio

