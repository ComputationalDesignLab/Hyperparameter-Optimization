HPO (Sequential)
=============================================

NNs excel in handling large, nonlinear datasets and multidimensional optimization problems, surpassing methods like kriging. However, NNs struggle with providing uncertainty estimation, crucial for exploitation in sequential sampling. Without uncertainty modeling, sequential sampling focused solely on exploitation risks convergence to local minima, hindering surrogate model refinement. To address this, Koratikere et al. introduced the efficient global optimization using NNs (EGONN) algorithm, which integrates sequential sampling with NN-based predictions and a secondary NN to estimate uncertainty.
  
.. toctree::
   :maxdepth: 3
 
   bohb_sequential
   imp_function_sequential
