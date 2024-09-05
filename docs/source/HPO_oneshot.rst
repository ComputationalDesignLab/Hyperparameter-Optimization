HPO (One-shot)
=========================

This section presents the results of applying HPO to SBO using one-shot methods. Various HPO techniques—**grid search**, **random search**, 
**Bayesian optimization**, **Hyperband**, and **Bayesian optimization Hyperband**—are tested to evaluate the impact of sample size on convergence and time efficiency. 
Sample sizes, incremented in steps of 10 from 10 to 50, assess convergence performance and computational cost, serving as a basis for later exploration of sequential sampling methods. 
Open-source machine-learning tools are employed, with ``Ax`` used for BO and ``HpBandSter`` for HB and BOHB.


.. toctree::
   :maxdepth: 3
 
   gs
   rs
   bo
   hb
   bohb
   imp_function_oneshot
