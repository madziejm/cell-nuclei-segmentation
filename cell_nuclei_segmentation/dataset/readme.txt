# Description of the Biostudies dataset S-BSST265
# Title: An annotated fluorescence image dataset for training nuclear segmentation methods
# The dataset is assigned with an open data license (CC0)
# Author: Florian Kromp, 15.04.2020
# Children's Cancer Research Institute, Vienna, Austria
# florian.kromp@ccri.at


The dataset contains 6 folders holding images and masks, a csv file describing each image and this readme.

Description of image_description.csv:
- contains information about each image: diagnosis, sample preparation, suggested train-/testset split, testset class, magnification, modality, software, mean bg signal, mean fg signal, signal/noise ratio, signal/noise class
- for each image, the mean background signal and the mean foreground signal were calculated, based on these measures the signal to noise ratio was calculated and based thereon, a classification into three classes was done.
- all images of the dataset were split into a training and testset
- the training set consists of images of ganglioneuroblastoma tissue sections, neuroblastoma bone marrow cytospin preparations, neuroblastoma tumor touchimprints, normal cell cytospin preparations and normal cells grown on slide;
  images of the training set were imaged with the same magnification per preparation, similar signal-to-noise ratios and using a non-confocal fluorescence microscope
- the test set consists of images of the same preparations as the training set images but is extended by images of neuroblastoma cell line cytospin preparations, a wilms tumor section and a neuroblastoma tumor section;
  test set images were imaged with varying conditions: signal-to-noise ratios, modalities (adding confocal images) and magnifications;
  based on these varying conditions, all images of the testset were classified into one out of 10 classes

Testset classes:
GNB-I ganglioneuroblastoma tissue sections
GNB-II ganglioneuroblastoma tissue sections with a low signal-to-noise ratio
NB-I neuroblastoma bone marrow cytospin preparations
NB-II neuroblastoma cell line preparations imaged with different magnifications
NB-III neuroblastoma cell line preparations imaged with LSM modalities
NB-IV neuroblastoma tumor touch imprints
NC-I normal cells cytospin preparations
NC-II normal cells cytospin preparations with low signal-to-noise ratio
NC-III normal cells grown on slide
TS other tissue sections (neuroblastoma, Wilms)

Folder description:
- rawimages: Raw nuclear images in TIFF format
- groundtruth: Annotated masks in TIFF format
- groundtruth_svgs: SVG-Files for each annotated masks and corresponding raw image in JPEG format
- singlecell_groundtruth: Groundtruth for randomly selected nuclei of the testset (25 nuclei per testset class, a subset of all nuclei of the testset classes; human experts can compete with this low number of nuclei per subset by calculating Dice coefficients between their annotations and the groundtruth annotations)
- visualized_groundtruth: Visualization of groundtruth masks in PNG format
- visualized_singlecell_groundtruth: Visualization of groundtruth for randomly selected nuclei in PNG format
