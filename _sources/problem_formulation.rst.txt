====================
Problem Formulation
====================

Basic elements of simulation-based engineering design optimization utilizing NNs as surrogate models are depicted in Fig. 3. 
It illustrates both one-shot and sequential sampling, which follow the same initial steps up to the construction of NNs. 
It begins with generating a design population, followed by physics-based simulations that evaluate corresponding outputs to form training data for the NNs. 
With this data, optimal HPs are determined to construct the NNs. 
These NNs replace the physics-based simulation to identify optimized designs in the one-shot method and discover new sample 
points based on the infill criterion in the sequential sampling. If a stopping condition is not reached, these new samples are 
integrated into the training dataset for future iterations.

.. image:: figures/workflow.PNG
  :width: 650
  :alt: Alternative text

.. centered::
	**Figure 3 : Flowchart of simulation-based engineering design optimization with neural networks for the one-shot method and sequential sampling.**

.. toctree::
   :maxdepth: 3
 
   hyperparameter
   hyperparameter_problem_formulation
   testcases
   
