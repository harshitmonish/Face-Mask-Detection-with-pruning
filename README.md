# Face-Mask-Detection
The aim of this project is to find an efficient COVID face mask detection model for Deployment.
In deep neural networks the computational cost for inference is higher and is proportional to the
number of users/queries. When these deep models are deployed on the cloud, edge devices, mobile
phones, etc. for various applications, low latency and less memory consumption are the key aspects
for inference to decrease the computational cost on the hardware. In order to reduce the compute demand
we can either optimize hardware and software stack or compress the model itself by reducing
the number of parameters. Since the latter looks more feasible and in comparison to optimizing the
hardware/software stack itself, We aim to explore different model compression techniques in this
project.

# Dataset
For this project the dataset that we are going to use MaskedFace-Net[2] Dataset. It consists of
137016 quality masked face images each of size 1024x1024. This dataset have three types of masked
face images i.e. Correctly Masked Face Dataset (CMFD), Incorrectly Masked Face Dataset (IMFD)
and their combination. The labels of this dataset consists of:
* Correctly masked
* Incorrectly masked
* * Uncovered Chin
* * Uncovered nose
* * Uncovered nose and mouth

First the image is classified as correctly masked(CMFD) or Incorrectly masked(IMFD) and then
IMFD is further classified as Uncovered chin, Uncovered node and Uncovered nose and mouth. An
image is labeled face mask correctly worn if the mask covers the nose, mouth, chin and incorrectly
if mask if just covering nose and mouth or mask covering mouth and chin or mask is under the
mouth.

# How to run:
* For Pruning code please refer to Pruning/FaceMaskPruning.ipynb
