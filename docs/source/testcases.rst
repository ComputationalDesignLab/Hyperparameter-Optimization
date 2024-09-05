Test Cases
========================

A two-dimensional Branin function is tested for the one-shot method, and a three-dimensional Hartmann function is tested for the sequential sampling method. In all cases, the Halton sequence is applied to ensure consistent starting points for all methods.

----------------------------------------
One-shot method - Branin function
----------------------------------------

The minimization of the two-dimensional Branin function is formulated as:

.. math::

    \min_{x_{d_1}, \text{ } x_{d_2}} f(x_{d_1},x_{d_2}) = \left[ x_{d_2} - \frac{5.1}{4\pi^2}x_{d_1}^2 + \frac{5}{\pi}x_{d_1} - 6 \right]^2 
    + 10\left[ 1 - \frac{1}{8\pi} \right] \cos(x_{d_1}) + 10,

with constraints :math:`x_{d1} \in [0, 8]` and :math:`x_{d2} \in [0, 8]`. The goal is to find a global minimum of 0.39 at :math:`(\pi, 2.275)`.

------------------------------------------------------------
Sequential sampling method - Hartmann function
------------------------------------------------------------

The minimization of the three-dimensional Hartmann function is expressed as:

.. math::

    \min_{\mathbf{x}_d} f(\mathbf{x}_d) = -\sum_{i=1}^{4} \alpha_i \exp \left( -\sum_{j=1}^{3} A_{ij}(x_{dj} - P_{ij})^2 \right),

where :math:`\alpha = [1.0, 1.2, 3.0, 3.2]^T`, and matrices :math:`\mathbf{A}` and :math:`\mathbf{P}` are defined as:

.. math::

    \mathbf{A} = \begin{bmatrix}
    3.0 & 10 & 30 \\
    0.1 & 10 & 35 \\
    3.0 & 10 & 30 \\
    0.1 & 10 & 35
    \end{bmatrix}, \quad 
    \mathbf{P} = 10^{-4} \begin{bmatrix}
    3689 & 1170 & 2673 \\
    4699 & 4387 & 7470 \\
    1091 & 8732 & 5547 \\
    381  & 5743 & 8828
    \end{bmatrix}.

The function is constrained by :math:`\mathbf{x}_{d_L} = (0,0,0)` and :math:`\mathbf{x}_{d_U} = (1,1,1)`, with one global minimum of -3.86 at :math:`(0.11, 0.56, 0.85)` and four local minima.
