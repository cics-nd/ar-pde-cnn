# Physics-Constrained Auto-Regressive Convolutional Neural Networks
Modeling the Dynamics of PDE Systems with Physics-Constrained Deep Auto-Regressive Networks [[JCP](https://doi.org/10.1016/j.jcp.2019.109056)][[ArXiv](https://arxiv.org/abs/1906.05747)]

[Nicholas Geneva](http://nicholasgeneva.com/), [Nicholas Zabaras](https://cics.nd.edu)

---
## 1D Viscous Burgers' Equation

This repository contains files pertaining to the modeling of the 1D Viscous Burgers' Equation.
The Burgers' equation is a fundamental PDE that arises in multiple areas ranging from fluid dynamics to traffic flow. It is most recognized for its characteristic shock formations.

![1D Viscous Burgers' Equation](../img/burger1d/burger_eq.png "1D Viscous Burgers' Equation")

![1D Viscous Burgers' Pred](../img/burger1d/burger_pred.png  "1D Viscous Burgers' Predictions")

---
## Quick Start

### Training AR-DenseED

```
python main.py --epochs 100
```

### Training BAR-DenseED

If training from scratch:
```
python main.py --epochs 200 --swag-start 101
```
If starting from pre-trained determinisitic model:
```
python main.py --epochs 200 --swag-start 101 --epoch-start 100
```
Additional details of program parameters can be found in `args.py`.

### Generating Testing Data

Generate your own testing data using FEM using `solver/fenics_burger1D.py` in Python. Note that you will require the [Fenics FEM package](https://fenicsproject.org/).
```
python fenics_burger1D.py --istart 0 --iend 200
```
Generate your own FDM data using `solver/fd_burger1D.py` in Python, however the finite element simulator must be used first to generate the initial states.
```
python fd_burger1D.py --istart 0 --iend 10
```
Alternatively, one can download the testing data for both FEM and FDM from [Notre Dame's Secure File Sharing Server](https://notredame.box.com/s/h3v7u85tn2a60gh7wqyeo17h1jvhuu0g).
Move the compressed folder to the `./solver` folder, and extract its contents with:
```
tar -xvzf ar_pde_1dBurgerData.tar.gz
```

### Creating Figures

The following scipts can be used to generate the figures seen in the paper. Pre-trained models are provided which are found in `./post/networks`. Testing data is required for these figures, see previous section for details on how to obtain the appropriate data. 

All of the following programs are found in the `./post` folder.

**Figure 9**: Plot four finite element simulations of the 1D Burgers' equation.
```
python plotFEMContour.py
```
**Figure 10**: Plot four test predictions using AR-DenseED.
```
python plotARContour.py
```
**Figure 11**: The average mean squared error (MSE) and energy squared error (ESE) as a function of time for a test set of 200 cases using AR-DenseED and predictive expectation of BAR-DenseED.
```
python plotMSE.py
```
**Figure 12**: Samples from the posterior of BAR-DenseED.
```
python plotBARSamples.py
```
**Figure 13**: Plot the predictive expectation and variance of BAR-DenseED for four test cases.
```
python plotBARContour.py
```
**Figure 14 and 15**: Plot the predictive profiles for a test case at several different times.
```
python plotProfiles.py
```

---
## Citation
Find this useful or like this work? Cite us with:
```latex
@article{geneva2019modeling,
  title = {Modeling the dynamics of {PDE} systems with physics-constrained deep auto-regressive networks},
  journal = {Journal of Computational Physics},
  pages = {109056},
  year = {2019},
  issn = {0021-9991},
  doi = {10.1016/j.jcp.2019.109056},
  url = {http://www.sciencedirect.com/science/article/pii/S0021999119307612},
  author = {Nicholas Geneva and Nicholas Zabaras}
}
```