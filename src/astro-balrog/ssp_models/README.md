Place the sMILES model file here, named exactly:

    smiles_bimodal.npz

It must contain the arrays: lam, templates, ages, metals, afes, slopes
(templates has shape (n_lam, n_ages, n_metals, n_afes, n_slopes)).

Any *.npz in this folder is included in the wheel/sdist when you build the package.
