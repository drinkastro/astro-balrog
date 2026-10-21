#import astro-balrog as BR
import fit_BR
from measure_BR import measure_BR
from plotting import cornerplot, plot_BR
from pathlib import Path
import numpy as np
from _init_ import path_to_models

def example_model():

    np.random.seed(123) #For reproducible results

    #Create model spectrum
    lib = np.load(path_to_models)
    lam, templates, ages, metals, afes = lib["lam"], lib["templates"], lib["ages"], lib["metals"], lib["afes"]
    wl = (lam > 3000) & (lam < 5000) #only consider wavelengths including Balmer lines to reduce computational time
    lam = lam[wl]

    #Considering a model at 1 Gyr, solar metallicity, solar [a/Fe], and MW-like IMF
    age = 1 #Gyr
    met = 0.06 #dex
    afe = 0. #solar scale
    t0 = templates[wl, ages==age, metals==met, afes==afe]

    #Adding noise to the model and simulate the error spectrum
    snr = 20
    galaxy = np.random.normal(t0, t0/snr)
    err = np.random.permutation(t0/snr)

    #Assuming the best-fitting age from SED fitting is consistent with the true age
    age_bf = 0.95
    age_err = 0.2

    #List of Balmer lines to fit and computing the ratio as (H_0 ... H_sumn-1)/(H_sumn + ... H_-1), where subscripts are the indices of the input list.
    blist = ['Hgamma_F', 'Hdelta_F', 'Hzeta', 'Htheta', 'Hiota', 'Hbeta', 'Heta'] #List of Balmer lines to fit
    sumn = 5

    br_obs, br_err = measure_BR(lam, galaxy, err, blist=blist, sumn=sumn)

    print("Input [M/H] = ", met, "dex")
    print("Measured BR: ", round(br_obs, 2), "+/-", round(br_err,2))

    plot_BR(br_obs, br_err, age_bf, age_err)


    samples = fit_BR.fit_BR(lam, galaxy, err, age_bf, age_err,
               blist=blist, sumn=sumn,
               afe_best=None, afe_err=None, nwalkers=30, nsteps=5000, discard=2000, thin=5)

    cornerplot(samples, truths=[age, met, afe])


if __name__ == '__main__':
    example_model()

