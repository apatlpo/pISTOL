pistol's documentation
============================

.. https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html

Equations of motion
-------------------

See this :doc:`equations`


Install
-------

We recommend conda for dependencies, see README on
`pistol github repository <https://github.com/apatlpo/pistol>`_

Tutorial
--------

To do ...

.. code:: bash

   mpirun -n 4 python analytical.py -mf -ksp_view -ksp_monitor -ksp_converged_reason

Profiling:

.. code:: bash

   mpirun -n 4 python -m cProfile -o output.prof uniform.py
   snakeviz output.prof


API
--------------------

.. toctree::
   :maxdepth: 2

   equations
   api/qgsolver

Indices and tables
--------------------

* :ref:`genindex`
* :ref:`search`
