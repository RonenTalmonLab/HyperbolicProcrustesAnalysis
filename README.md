# Hyperbolic Procrustes Analysis Using Riemannian Geometry
The code in this repository creates the figures presented in this article: 
https://proceedings.neurips.cc/paper/2021/file/2ed80f6311c1825feb854d78fa969d34-Paper.pdf

Please notice that in order to apply the code to the data sets, they need to be downloaded first from the following specified links. 
The code was developed and tested in Python 3.8.
  
## Demo - Simulations 
The script 'Main.py' generates the discrepancies of baseline, HPA, and RT for the simulated examples, reported in Figure 2.

## Batch correction for bioinformatics dataset and domain adaptation for digit datasets

### Data
* BC: download METABRIC at the link https://www.cbioportal.org/study/summary?id=brca_metabric and TCGA at the link https://www.cbioportal.org/study/summary?id=brca_tcga_pub
* LC: download the data at the link https://ascopubs.org/doi/suppl/10.1200/JCO.2005.05.1748
* CyTOF: access the denoised data at the link https://github.com/ushaham/BatchEffectRemoval
* Digits: download MNIST at http://yann.lecun.com/exdb/mnist/ and USPS at https://www.kaggle.com/bistaumanga/usps-dataset

### Hyperbolic representation 
Run all the batches/domains with the code at the link https://github.com/facebookresearch/poincare-embeddings

Set the hyperparameters with manifold lorentz, learning rate 0.001, train threads 2, and batch size 20.

## Demo - HPA

```python
from manifold_func import *

# data is the obtained hyperbolic representation in the Lorentz model
# the data type is a list that each list item represents the data from one batch/domain 
HPA_data = HPA_align_tan(data)
```


