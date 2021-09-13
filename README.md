# CMIR-NET-A-deep-learning-based-model-for-cross-modal-retrieval-in-remote-sensing
[Paper](https://doi.org/10.1016/j.patrec.2020.02.006) | [TensorFlow](https://www.tensorflow.org/) | [PPT](https://github.com/ushasi/CMIR-NET-A-deep-learning-based-model-for-cross-modal-retrieval-in-remote-sensing/blob/master/CMIRNet_IEEE_Young_researchers_conclave.pdf)

We address the problem of cross-modal information retrieval in the domain of remote sensing. In particular, we are interested in two application scenarios: i) cross-modal retrieval between panchromatic (PAN) and multispectral imagery, and ii) multi-label image retrieval between very high resolution (VHR) images and speech-based label annotations. These multi-modal retrieval scenarios are more challenging than the traditional uni-modal retrieval approaches given the inherent differences in distributions between the modalities. However, with the increasing availability of multi-source remote sensing data and the scarcity of enough semantic annotations, the task of multi-modal retrieval has recently become extremely important. In this regard, we propose a novel deep neural network-based architecture that is considered to learn a discriminative shared feature space for all the input modalities, suitable for semantically coherent information retrieval. Extensive experiments are carried out on the benchmark large-scale PAN - multispectral DSRSID dataset and the multi-label UC-Merced dataset. Together with the Merced dataset, we generate a corpus of speech signals corresponding to the labels. Superior performance with respect to the current state-of-the-art is observed in all the cases.


<img src=cmir.jpg alt="Pipeline of the overall network" width="700">


In this work we have experimented on {Panchromatic satellite data, Multispectral satellite data} and {RGB image, speech} cross data. To implement these, go to the corrosponding sub-folder:
1. Run the pretraining codes for both the modalities and save the features in a .mat file.
2. Run Unified_XY.py. This saves a unified.mat file which stores the latent-space features for both the modalities.
3. Use the MATLAB KNNcode.m to find the top $k$-nn retrieved data (using simple Eucledean distance).




### Paper

*   If you find this code useful, please cite the following paper:

```
@InProceedings{Chaudhuri_2020_cmirnet,
author = {Chaudhuri, Ushasi and Banerjee, Biplab and Bhattacharya, Avik and Datcu, Mihai},
title = {CMIR-NET : A deep learning based model for cross-modal retrieval in remote sensing},
booktitle = {In Pattern Recognition Letters},
month = {March},
year = {2020},
volume = {131},
pages = {456-462},
} 
