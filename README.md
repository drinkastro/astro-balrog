# astro-balrog

<p align="center">
  <img src="logo/balrog_logo.png" alt="Project logo" width="200">
</p>


The Balmer Absorption Line Ratio fittinG (BALROG) code performs a fit of the ratio of the Balmer absorption lines, measured from a galaxy spectrum, to infer the stellar metallicity. The fit performs a simple MCMC that compares the observed Balmer Ratio with the one inferred from the models. BALROG is currently designed to work only with the sMILES stellar population library (assuming a bimodal, MW-like IMF). 

To install BALROG, you just need to download the zip file from github. Before running the code, it is necessary to download the .npz file containing the sMILES models. You can find the link to the zenodo repistory containing the models in the "ssp_models/download_smiles.txt" file. Download the file and move it to the ssp_models directory in your computer. 

Please cite Bevacqua+ submitted if you use this code.
