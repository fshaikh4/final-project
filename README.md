# Prediction of mental status from structural MRI and demographics via multi-input deep learning
#### BIOF 509 Final Project (Spring 2020)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fshaikh4/final-project/master?filepath=final-project.ipynb)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fshaikh4/final-project/blob/master/final-project.ipynb)

#### Tools utilized for this project

 - [Neurodocker](https://hub.docker.com/r/repronim/neurodocker/)
 - [Singularity](https://sylabs.io/about-us)
 - Several tools from the [nipy ecosystem](https://nipy.org/index.html#):
   - [Nipype](https://nipype.readthedocs.io/en/latest/)
   - [NiBabel](https://nipy.org/nibabel/)
   - [nilearn](https://nilearn.github.io/#)
 - [FSL: FMRIB Software Library](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)
   - [FLIRT: FMRIB's Linear Image Registration Tool](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)
   - [MNI152 template brain](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases)


#### Additional references

 - [neuroimage-tensorflow](https://github.com/corticometrics/neuroimage-tensorflow)
 - [An Introduction to Biomedical Image Analysis with TensorFlow and DLTK](https://blog.tensorflow.org/2018/07/an-introduction-to-biomedical-image-analysis-tensorflow-dltk.html)
 - [Keras: Multiple Inputs and Mixed Data](https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/)

## Preparatory steps

#### Setup containerized workspace

A Singularity recipe file (`biof509.final`) was generated via [Neurodocker](https://hub.docker.com/r/repronim/neurodocker/) with all relevant required software for this project. Please ensure you have [Singularity](https://sylabs.io/about-us) installed to initialize and work inside of the container.

The Singularity recipe file was compiled into the Singularity container via [SingularityHub](https://singularityhub.org/) [[more information here](https://singularityhub.github.io/singularityhub-docs/docs/introduction)].

#### Cloning `git` repo

Please feel free to clone this repository locally via `git clone https://github.com/fshaikh4/final-project.git`, or access the relevant files using the above-linked binder or Colab buttons.

*NOTE: some preprocessing steps require use of a Neurodocker-created containerized workspace. As such, it is suggested to clone the repo inside of the containerized workspace, after building the workspace as described above.*

#### Data source

Data for this project were taken from the [Dallas Lifespan Bran Study (DLBS)](http://fcon_1000.projects.nitrc.org/indi/retro/dlbs.html), and are publicly available online (as `.tar.gz` files):
- [Cognitive data](ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/dlbs_content/dlbs_cogdata.tar.gz)
- [Neuroimaging data](ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/dlbs_content/dlbs_imaging.tar.gz)
   - [Anatomical scan parameters](ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/dlbs_content/dlbs_scan_params_anat.pdf)
   - [PET scan parameters](ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/dlbs_content/dlbs_scan_params_pet.pdf)
- [Genetic data](ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/dlbs_content/dlbs_genetics.tar.gz)

An initial attempt was made to retrieve relevant files via python `ftlib` operations. However due to memory, bandwidth, and/or possibly other unknown constraints, this attempt was not succesful. To move forward, relevant data were instead downloaded via a web browser and saved into a `./data/` dir (not included directly in this repo).

## Problem statement

**Can we predict an individual's Mini-Mental State Examination (MMSE) score, given their structural MRI and demographic information?**

#### Predictors
 - T1-weighted Structural MRI images
 - APOE4 carrier status
 - Age
 - Biological sex
 - Years of education

#### Machine learning task

 Since we are predicting MMSE, a numerical feature, our task is **regression.**
 
#### Relevant literature

 - [3D-Deep Learning Based Automatic Diagnosis of Alzheimer's Disease with Joint MMSE Prediction Using Resting-State fMRI.](https://doi.org/10.1007/s12021-019-09419-w)
 - [High-dimensional pattern regression using machine learning: from medical images to continuous clinical variables.](https://doi.org/10.1016/j.neuroimage.2009.12.092)
 - [Deep learning based low-cost high-accuracy diagnostic framework for dementia using comprehensive neuropsychological assessment profiles.](https://doi.org/10.1186/s12877-018-0915-z)

#### Approach

I developed a multi-input neural network that accepts both image files as well as tabular data as inputs, and that provides regression output to predict an individual participant's MMSE score.

I utilized a **60:20:20 training-validation-testing split** in this project, where the validation set was used to tune model "hyperparameters" such as network architecture, number of epochs, loss function, etc.

## Data preprocessing steps

#### Non-image data

#### Image data

## Model comparison

Due to the nature of the task (regression) and the nature of predictions (a predicted *score* of a neuropsychological exam), model fit was reported in terms of **mean absolute error (MAE),** where a lower MAE represented more accurate predictions of MMSE score.

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{MAE}&space;=&space;\frac{1}{n}&space;\cdot&space;\sum\limits^{n}_{i=1}&space;|y_i&space;-&space;\hat{y_i}|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{MAE}&space;=&space;\frac{1}{n}&space;\cdot&space;\sum\limits^{n}_{i=1}&space;|y_i&space;-&space;\hat{y_i}|" title="\text{MAE} = \frac{1}{N} \cdot \sum\limits^{N}_{i=1} |y_i - \hat{y_i}|" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n" title="n" /></a> represents the total number of participants (and thus predictions), and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y_i}" title="\hat{y_i}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=y_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" /></a>  are the predicted and actual (respectively) MMSE for the <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>th participant.

#### Na&iuml;ve model

To have a reference point of comparison against the sophisticated multi-input deep learning model against, I first ran all non-image predictors through a regularized elastic net regression with hyperparameters tuned via grid search. The model yielded the following results.s

## Network architecture

## Results

## Future directions
