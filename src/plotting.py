import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from .measure import measure_BR
from ._paths import path_to_models


def cornerplot(samp, truths = None):

    lab = ['Age (Gyr)', '[M/H] (dex)', '[a/Fe]']
    fig = corner.corner(samp, labels=lab, quantiles=[0.16, 0.5, 0.84], show_titles=True, density=True,
                            truths=truths, truth_color='k')

    plt.show()

def plot_BR(br_obs=None, br_obs_err=None, age_bf=None, age_err=None, afe=0.0, imf=1.3,
           age_range=[0.1, 3], met_range = [-1.79, 0.26], title=None,
                      blist = ['Hgamma_F', 'Hdelta_F', 'Hzeta', 'Htheta', 'Hiota', 'Hbeta', 'Heta'], sumn = 5,
                      savefig_path=None, cmap='plasma'):

    lib = np.load(path_to_models)
    lam, templates, ages, metals, afes, imfs = lib["lam"], lib["templates"], lib["ages"], lib["metals"], lib["afes"], lib["slopes"]

    if age_range is not None:
        wa = (ages >= age_range[0]) & (ages <= age_range[1])
        ages = ages[wa]
        templates = templates[:, wa, :, :, :]

    if met_range is not None:
        wm = (metals >= met_range[0]) & (metals <= met_range[1])
        metals = metals[wm]
        templates = templates[:, :, wm, :, :]

    templates = templates[:, :, :, afes==afe, imfs == imf] #Fixed a/Fe and imf slope

    param_shape = (len(ages), len(metals))

    ratios = np.zeros(param_shape)
    for a,age in enumerate(ages):
        for m, met in enumerate(metals):
            wa = ages == age
            wm = metals == met
            t0 = templates[:, wa, wm, 0].ravel()

            ratios[a, m] = measure_BR(lam, t0, blist=blist, sumn=sumn)[0]


    fig = plt.figure(figsize=(7,8))



    norm = matplotlib.colors.Normalize(vmin=metals.min(), vmax=metals.max())
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    color = np.array([(mapper.to_rgba(v)) for v in metals])
    for i in range(len(color)):
        plt.plot(ages, ratios[:,i], 'o-', color=color[i], label=str(round(metals[i],2)), lw=3)

    if br_obs is not None:
       plt.errorbar(age_bf, br_obs, xerr=age_err, yerr=br_obs_err, fmt='o', markersize=10, ecolor='k', elinewidth=2, capsize=3, capthick=3, mec='k', mfc='k')


    plt.xlabel('Age (Gyr)', fontsize=20)

    ylab = "("+"+".join(blist[:sumn])+") / (" + "+".join(blist[sumn:]) + ")"
    plt.ylabel(str(ylab), fontsize=20)

    plt.tick_params(labelsize=16)
    if title is None: plt.title(title, fontsize=23)
    plt.legend(loc='lower left', title='[Z/H] (dex)', title_fontsize=18, fontsize=16)
    plt.tight_layout()
    plt.minorticks_on()
    if savefig_path is not None: plt.savefig(savefig_path+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)

    plt.show()

if __name__ == '__main__':
    plot_BR(br_obs=2, br_obs_err=0.05, age_bf=1, age_err=0.1)

