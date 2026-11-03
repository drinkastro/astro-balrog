import numpy as np
from .fit import fit_BR
from .measure import measure_BR
from .plotting import cornerplot, plot_BR
from ._paths import path_to_models

def model_spectrum():

    np.random.seed(123) #For reproducible results

    #Create model spectrum
    lib = np.load(path_to_models)
    lam, templates, ages, metals, afes = lib["lam"], lib["templates"], lib["ages"], lib["metals"], lib["afes"]
    wl = (lam > 3000) & (lam < 5000) #only consider wavelengths including Balmer lines to reduce computational time
    lam = lam[wl]

    #Considering a model having 1 Gyr, solar metallicity, solar [a/Fe], and MW-like IMF
    age = 1 #Gyr
    met = 0.06 #dex
    afe = 0. #solar scale
    t0 = templates[wl, ages==age, metals==met, afes==afe]

    #Adding noise to the model and simulate the error spectrum
    snr = 20
    galaxy = np.random.normal(t0, t0/snr) #Adding Gaussian noise
    err = np.random.permutation(t0/snr) #Creating error spectrum to fit by permutating the noise spectrum added to the model, to avoid fitting the exact error spectrum

    #Assuming the best-fitting age from SED fitting is consistent with the true age
    age_bf = 0.95
    age_err = 0.2

    #List of Balmer lines to fit and computing the ratio as (H_0 ... H_sumn-1)/(H_sumn + ... H_-1), where subscripts are the indices of the input list.
    blist = ['Hgamma_F', 'Hdelta_F', 'Hzeta', 'Htheta', 'Hiota', 'Hbeta', 'Heta'] #List of Balmer lines to fit
    sumn = 5

    br_obs, br_err = measure_BR(lam, galaxy, err, blist=blist, sumn=sumn)

    print("Measured BR: ", round(br_obs, 2), "+/-", round(br_err,2))

    plot_BR(br_obs, br_err, age_bf, age_err)
    plt.show()


    samples = fit_BR(lam, galaxy, err, age_bf, age_err,
               blist=blist, sumn=sumn,
               afe_best=None, afe_err=None, nwalkers=30, nsteps=1000, discard=100, thin=5)

    met16, met50, met84 = np.percentile(samples[:, 1], [16, 50, 84])
    met_le, met_ue = met50 - met16, met84 - met50

    print(rf"Best-fitting [M/H]: {met50:.2f}^{{+{met_ue:.2f}}}_{{-{met_le:.2f}}} dex (Input: {met:.2f} dex)")

    cornerplot(samples, truths=[age, met, afe])
    plt.show()

def observed_spectrum():
    np.random.seed(123)  # For reproducible results

    #Upload galaxy spectrum
    lam, galaxy, err = np.loadtxt('./data/legac_M2_133501.ascii').T

    z = 0.7292
    masknan = np.where(~np.logical_or(np.isnan(galaxy), np.isnan(err)))
    err, galaxy, lam = err[masknan], galaxy[masknan], lam[masknan]
    lam /= (1 + z) #rest-frame

    blist = ['Hgamma_F', 'Hdelta_F', 'Hzeta', 'Hbeta']
    sumn = 3
    age_bf, age_err = 1.77, 0.12 #best-fit from pPXF (see Bevacqua+26)
    sigma_gal = 203 #km/s


    met_fif, afe_fif = -0.29, 0.14 #Metallicity and [a/Fe] estimated from the FIF (see Bevacqua+)


    br_obs, br_err = measure_BR(lam, galaxy, err, blist=blist, sumn=sumn)

    print("Measured BR: ", round(br_obs, 2), "+/-", round(br_err, 2))

    afe_smiles = 0.2 #Only needed to plot the models

    plot_BR(br_obs=br_obs, br_obs_err=br_err, age_bf=age_bf, age_err=age_err, blist = blist, sumn = sumn, afe_smiles=afe_smiles)
    plt.show()

    samples = fit_BR(lam, galaxy, err, age_bf, age_err, sigma_gal=sigma_gal, blist=blist, sumn=sumn, nsteps=1000, discard=100, thin=5)

    met16, met50, met84 = np.percentile(samples[:, 1], [16, 50, 84])
    met_le, met_ue = met50 - met16, met84 - met50

    print(rf"Best-fitting [M/H]: {met50:.2f}^{{+{met_ue:.2f}}}_{{-{met_le:.2f}}} dex (FIF: {met_fif:.2f} dex)")

    cornerplot(samples, truths=[age_bf, met_fif, afe_fif])
    plt.show()


if __name__ == '__main__':
    model_spectrum()
    observed_spectrum()
