# Conditional deep surrogate models for stochastic, high-dimensional, and multi-fidelity systems
We present a probabilistic deep learning methodology that enables the construction of predictive data-driven surrogates for stochastic systems. Leveraging recent advances in variational inference with implicit distributions, we put forth a statistical inference framework that enables the end-to-end training of surrogate models on paired input-output observations that may be stochastic in nature, originate from different information sources of variable fidelity, or be corrupted by complex noise processes. The resulting surrogates can accommodate high-dimensional inputs and outputs and are able to return predictions with quantified uncertainty. The effectiveness our approach is demonstrated through a series of canonical studies, including the regression of noisy data, multi-fidelity modeling of stochastic processes, and uncertainty propagation in high-dimensional dynamical systems.

This paper was published on Computational Mechanics.

- Yibo, Yang, and Paris Perdikaris. "[Conditional deep surrogate models for stochastic, high-dimensional, and multi-fidelity systems.](https://link.springer.com/article/10.1007/s00466-019-01718-y)" Computational Mechanics volume 64, pages417–434(2019).

For the high dimensional Burgers example, the data is too large that we are not able to provide here, but you can find them in: https://drive.google.com/file/d/1n4a5Bivt6INq2xHSlVByZaJn-lB0mImK/view?usp=sharing


## Citation
```
@Article{Yang2019,
author="Yang, Yibo
and Perdikaris, Paris",
title="Conditional deep surrogate models for stochastic, high-dimensional, and multi-fidelity systems",
journal="Computational Mechanics",
year="2019",
month="May",
day="21",
abstract="We present a probabilistic deep learning methodology that enables the construction of predictive data-driven surrogates for stochastic systems. Leveraging recent advances in variational inference with implicit distributions, we put forth a statistical inference framework that enables the end-to-end training of surrogate models on paired input--output observations that may be stochastic in nature, originate from different information sources of variable fidelity, or be corrupted by complex noise processes. The resulting surrogates can accommodate high-dimensional inputs and outputs and are able to return predictions with quantified uncertainty. The effectiveness our approach is demonstrated through a series of canonical studies, including the regression of noisy data, multi-fidelity modeling of stochastic processes, and uncertainty propagation in high-dimensional dynamical systems.",
issn="1432-0924",
doi="10.1007/s00466-019-01718-y",
url="https://doi.org/10.1007/s00466-019-01718-y"
}

```
