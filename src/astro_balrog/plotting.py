import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from .measure import measure_BR
from ._paths import path_to_models


def cornerplot(samp, truths = None):
    """
    Simple cornerplot
    :param samp: samples from the emcee fit    
    :return: fig
    """
    lab = ['Age (Gyr)', '[M/H] (dex)', '[a/Fe]']
    fig = corner.corner(samp, labels=lab, quantiles=[0.16, 0.5, 0.84], show_titles=True, density=True,
                            truths=truths, truth_color='r')

    return fig

def plot_BR(br_obs=None, br_obs_err=None, age_bf=None, age_err=None, afe_smiles=0.0,
           age_range=[0.1, 3], met_range = [-1.79, 0.26],
                      blist = ['Hgamma_F', 'Hdelta_F', 'Hzeta', 'Htheta', 'Hiota', 'Hbeta', 'Heta'], sumn = 5):
    """
    Plot the Balmer ratio predicted by models and compare with observation.

    :param br_obs, br_obs_err: observed balmer ratio and relative error. If None, only the model predictions will be plotted.
    :param age_bf, age_err: best-fitting age and relative error.
    :param afe_smiles: [a/Fe] value assumed for the models, must be a value [-0.2, 0.0, 0.2, 0.4, 0.6] (default is solar scale)
    :param age_range: range of ages to be plotted on the x-axis
    :param met_range: range of metallicities to be plotted as colored curves
    :param blist: list of balmer lines from which the BR is calculated
    :param sumn: number of balmer lines to be summed

    :return: fig, axes
    """
    
    lib = np.load(path_to_models)
    lam, templates, ages, metals, afes = lib["lam"], lib["templates"], lib["ages"], lib["metals"], lib["afes"]
    if age_range is not None:
        wa = (ages >= age_range[0]) & (ages <= age_range[1])
        ages = ages[wa]
        templates = templates[:, wa, :, :]

    if met_range is not None:
        wm = (metals >= met_range[0]) & (metals <= met_range[1])
        metals = metals[wm]
        templates = templates[:, :, wm, :]

    templates = templates[:, :, :, afes==afe_smiles] #Fixed a/Fe

    param_shape = (len(ages), len(metals))

    ratios = np.zeros(param_shape)
    for a,age in enumerate(ages):
        for m, met in enumerate(metals):
            wa = ages == age
            wm = metals == met
            t0 = templates[:, wa, wm, 0].ravel()

            ratios[a, m] = measure_BR(lam, t0, blist=blist, sumn=sumn)[0]


    fig = plt.figure(figsize=(7,8))
    ax = plt.gca()



    norm = matplotlib.colors.Normalize(vmin=metals.min(), vmax=metals.max())
    mapper = cm.ScalarMappable(norm=norm, cmap="plasma")
    color = np.array([(mapper.to_rgba(v)) for v in metals])
    for i in range(len(color)):
        ax.plot(ages, ratios[:,i], 'o-', color=color[i], label=str(round(metals[i],2)), lw=3)

    if br_obs is not None:
       ax.errorbar(age_bf, br_obs, xerr=age_err, yerr=br_obs_err, fmt='o', markersize=10, ecolor='k', elinewidth=2, capsize=3, capthick=3, mec='k', mfc='k')


    ax.set_xlabel('Age (Gyr)', fontsize=20)
    texnames = []
    for name in blist:
        texnames.append("H$\\" + name[1:] + "$")



    ylab = "("+"+".join(texnames[:sumn])+") / (" + "+".join(texnames[sumn:]) + ")"
    ax.set_ylabel(str(ylab), fontsize=20)

    ax.tick_params(labelsize=16)
    ax.legend(loc='lower left', title='[Z/H] (dex)', title_fontsize=18, fontsize=16)
    ax.minorticks_on()
    ax.set_title(rf'[$\alpha$/Fe] = {afe_smiles:.1f} dex', fontsize=16)

    return fig, ax

