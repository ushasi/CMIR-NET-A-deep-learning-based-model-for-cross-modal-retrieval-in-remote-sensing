# CMIR-NET-A-deep-learning-based-model-for-cross-modal-retrieval-in-remote-sensing
[Paper](https://doi.org/10.1016/j.patrec.2020.02.006) | [TensorFlow](https://www.tensorflow.org/)

We address the problem of cross-modal information retrieval in the domain of remote sensing. In particular, we are interested in two application scenarios: i) cross-modal retrieval between panchromatic (PAN) and multispectral imagery, and ii) multi-label image retrieval between very high resolution (VHR) images and speech-based label annotations. These multi-modal retrieval scenarios are more challenging than the traditional uni-modal retrieval approaches given the inherent differences in distributions between the modalities. However, with the increasing availability of multi-source remote sensing data and the scarcity of enough semantic annotations, the task of multi-modal retrieval has recently become extremely important. In this regard, we propose a novel deep neural network-based architecture that is considered to learn a discriminative shared feature space for all the input modalities, suitable for semantically coherent information retrieval. Extensive experiments are carried out on the benchmark large-scale PAN - multispectral DSRSID dataset and the multi-label UC-Merced dataset. Together with the Merced dataset, we generate a corpus of speech signals corresponding to the labels. Superior performance with respect to the current state-of-the-art is observed in all the cases.


<img src=image/cmir.jpg alt="Pipeline of the overall network" width="700">

If you find this code useful, please cite the following paper:

CMIR-NET : A deep learning based model for cross-modal retrieval in remote sensing

Ushasi Chaudhuri, Biplab Banerjee, Avik Bhattacharya, Mihai Datcu. In Pattern Recognition Letters
Volume 131, March 2020, Pages 456-462. (https://doi.org/10.1016/j.patrec.2020.02.006)




In this work we have experimented on {Panchromatic satellite data, Multispectral satellite data} and {RGB image, speech} cross data. To implement these, go to the corrosponding sub-folder:
1. Run the pretraining codes for both the modalities and save the features in a .mat file.
2. Run Unified_XY.py. This saves a unified.mat file which stores the latent-space features for both the modalities.
3. Use the MATLAB KNNcode.m to find the top $k$-nn retrieved data (using simple Eucledean distance).
