Equations of motions
===========================

Continuous form
---------------

Boussinesq, adiabatic, hydrostatic and nonlinear shallow water equations for conservation of momentum:

.. math::

   \partial_t \mathbf{u}_n(x,y) + (\zeta_n + f) \mathbf{k}\times\mathbf{u}_n  = -\nabla \Big \{ M_n + \frac{1}{2} |\mathbf{u}_n|^2 + g \Pi \Big \}
               + \mathbf{H}_n + \mathbf{V}_n

for :math:`n=0,\cdotsN-1` and where :math:`f` is the Coriolis frequency and :math:`\zeta_n = \mathbf{k}\cdot (\nabla \times \mathbf{u}_n )`
is the relative vorticity, :math:`M_n` is the perturbation Montgomery potential.
The latter is given by:

.. math::

   M_0 = g_{1/2} \eta
   M_n = M_{n-1} + g_{n-1/2} \eta

