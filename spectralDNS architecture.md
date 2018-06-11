# spectralDNS architecture


## `demo/`

Contains executable scripts which may be run this way:
```
mpirun -np 4 python TG.py --M 6 6 6 --precision single --dealias '3/2-rule' NS 
```
	
Executable scripts have the following structure (from `demos/MKM.py`):
	
~~~python
from spectralDNS import config, get_solver, solve
...
def initOS(OS, U, X, t=0.): # not sure what this is used for
def initialize(solver, context):
def init_from_file(filename, solver, context):
def set_Source(Source, Sk, ST, FST, **context):
def dx(u, FST):
    """Compute integral of u over domain"""
def update(context):
class Stats(object):
    def __init__(self, U, comm, fromstats="", filename=""):
    def create_statsfile(self):
    def get_stats(self, tofile=True):
    def reset_stats(self):
    def fromfile(self, filename="stats"):
	
if __name__ == "__main__":
    solver = get_solver(update=update, mesh="channel")
    context = solver.get_context()
    initialize(solver, context)
    #init_from_file("KMM665c.h5", solver, context)
    set_Source(**context)
    solver.stats = Stats(context.U, solver.comm, filename="KMMstatsq")
    solve(solver, context)
~~~


## `spectralDNS/__utils__.py`

Contains several core methods:
	
~~~python
def get_solver(update=None,
               regression_test=None,
               additional_callback=None,
               mesh="triplyperiodic", parse_args=None):
    """Return solver based on global config (see spectralDNS/config.py)

    args:
        update               Update function called on each timestep.
                             Typically used for plotting or computing
                             intermediate results
        regression_test      Function called at the end of simulations.
                             Typically used for checking results
        additional_callback  Function used by some integrators that require
                             additional callback
        mesh                 Type of problem ('triplyperiodic',
                                              'doublyperiodic',
                                              'channel')
        parse_args           Used to specify arguments to config.
                             If parse_args is None then Commandline arguments
                             are used.

    global args:
        config               See spectralDNS/config.py for details.

    """
 	...
    return solver


def solve(solver, context):
    """Generic solver for spectralDNS

    args:
        solver       The solver (e.g., NS or VV) module
        context      The solver's context

    global args:
        params       Dictionary (config.params) of parameters
                     that control the integration.
                     See spectralDNS.config.py for details
    """

    solver.timer = solver.Timer()
    params = solver.params

    solver.conv = solver.getConvection(params.convection)

    integrate = solver.getintegrator(context.dU, # rhs array
                                     context.u,  # primary variable
                                     solver,
                                     context)

    dt_in = params.dt

    while params.t + params.dt <= params.T+1e-12:

        u, params.dt, dt_took = integrate()

        params.t += dt_took
        params.tstep += 1

        solver.update(context)

        context.hdf5file.update(params, **context)

        solver.timer()

        if not solver.profiler.getstats() and params.make_profile:
            #Enable profiling after first step is finished
            solver.profiler.enable()

        if solver.end_of_tstep(context):
            break

    params.dt = dt_in

    solver.timer.final(params.verbose)

    if params.make_profile:
        solver.results = solver.create_profile(solver.profiler)

    solver.regression_test(context)

    context.hdf5file.close()
~~~



## `spectralDNS/solvers/`

Contains solvers, `KMM.py` for example:

~~~python
from shenfun.spectralbase import inner_product
from shenfun.la import TDMA
from shenfun import TensorProductSpace, Array, TestFunction, TrialFunction, \
    VectorTensorProductSpace, div, grad, Dx, inner, Function, Basis
from shenfun.chebyshev.la import Helmholtz, Biharmonic

from .spectralinit import *
from ..shen.Matrices import BiharmonicCoeff, HelmholtzCoeff
from ..shen import LUsolve
from ..shen.la import Helmholtz as old_Helmholtz

def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    K0 = Basis(params.N[1], 'F', domain=(0, params.L[1]), dtype='D')
    ...
    FST = TensorProductSpace(comm, (ST, K0, K1), **{'threads':params.threads, 'planner_effort':params.planner_effort["dct"]})    # Dirichlet    
    # Padded ...
    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)
    float, complex, mpitype = datatypes("double")
    # Mesh variables ...
    # Solution variables ...
    # primary variable ...
    # Set Nyquist frequency to zero on K that is used for odd derivatives in nonlinear terms ...
    work = work_arrays()
    nu, dt, N = params.nu, params.dt, params.N
    # Collect all matrices ...
    ## Collect all linear algebra solvers
    hdf5file = KMMWriter({"U":U[0], "V":U[1], "W":U[2]},
                         chkpoint={'current':{'U':U}, 'previous':{'U':U0}},
                         filename=params.solver+".h5",
                         mesh={"x": x0, "y": x1, "z": x2})
    return config.AttributeDict(locals())

class KMMWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        
assert params.precision == "double"

def end_of_tstep(context):
    """Function called at end of time step.
    If returning True, the while-loop in time breaks free. Used by adaptive
    solvers to modify the time stepsize. Used here to rotate solutions.
    """

def get_velocity(U, U_hat, VFS, **context):
    return U

def set_velocity(U_hat, U, VFS, **context):
    return U_hat

def get_curl(curl, U_hat, g, work, FST, SB, ST, Kx, **context):

def get_convection(H_hat, U_hat, g, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la, **context):
    """Compute convection from context"""
    conv_ = getConvection(params.convection)
    H_hat = conv_(H_hat, U_hat, g, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la)
    return H_hat

def get_pressure(context, solver):
    return p-uu+3./16.

def Div(U, U_hat, FST, K, work, la, mat, **context):
    return dudx+dvdy+dwdz

def Cross(c, a, b, FSTp, work):

@optimizer
def mult_K1j(K, a, f):

def compute_curl(c, u_hat, g, K, FCTp, FSTp, FSBp, work):

def compute_derivatives(duidxj, u_hat, FST, FCT, FSB, K, la, mat, work):
    return duidxj

def standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp, FSBp, FCTp, work,
                       mat, la):

def divergenceConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp, FSBp, FCTp, work,
                         mat, la, add=False):

def getConvection(convection):

    if convection == "Standard":

        def Conv(rhs, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la):
            u_dealias = work[((3,)+VFSp.backward.output_array.shape, float, 0)]
            u_dealias = VFSp.backward(u_hat, u_dealias)
            rhs = standardConvection(rhs, u_dealias, u_hat, K, VFSp, FSTp,
                                     FSBp, FCTp, work, mat, la)
            rhs[:] *= -1
            return rhs
	...
    Conv.convection = convection
    return Conv

@optimizer
def assembleAB(H_hat0, H_hat, H_hat1):
    H_hat0[:] = 1.5*H_hat - 0.5*H_hat1
    return H_hat0

@optimizer
def add_linear(rhs, u, g, work, AB, AC, SBB, ABB, BBB, nu, dt, K2, K4):
    return rhs

#@profile
def ComputeRHS(rhs, u_hat, g_hat, solver,
               H_hat, H_hat1, H_hat0, VFSp, FSTp, FSBp, FCTp, work, Kx, K2,
               K4, hv, hg, mat, la, **context):
    """Compute right hand side of Navier Stokes

    args:
        rhs         The right hand side to be returned
        u_hat       The FST of the velocity at current time
        g_hat       The FST of the curl in wall normal direction
        solver      The current solver module

    Remaining args are extracted from context

    """
    # Nonlinear convection term at current u_hat
    H_hat = solver.conv(H_hat, u_hat, g_hat, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la)

    # Assemble convection with Adams-Bashforth at time = n+1/2
    H_hat0 = solver.assembleAB(H_hat0, H_hat, H_hat1)

    # Assemble hv, hg and remaining rhs
    ...
    
    rhs = solver.add_linear(rhs, u_hat[0], g_hat, work, mat.AB, mat.AC, mat.SBB,
                            mat.ABB, mat.BBB, params.nu, params.dt, K2, K4)

    return rhs

@optimizer
def compute_vw(u_hat, f_hat, g_hat, K_over_K2):
    u_hat[1] = -1j*(K_over_K2[0]*f_hat - K_over_K2[1]*g_hat)
    u_hat[2] = -1j*(K_over_K2[1]*f_hat + K_over_K2[0]*g_hat)
    return u_hat

#@profile
def solve_linear(u_hat, g_hat, rhs,
                 work, la, mat, K_over_K2, H_hat0, U_hat0, Sk, **context):
    return u_hat, g_hat

def integrate(u_hat, g_hat, rhs, dt, solver, context):
    """Regular implicit solver for KMM channel solver"""
    rhs[:] = 0
    rhs = solver.ComputeRHS(rhs, u_hat, g_hat, solver, **context)
    u_hat, g_hat = solver.solve_linear(u_hat, g_hat, rhs, **context)
    return (u_hat, g_hat), dt, dt

def getintegrator(rhs, u0, solver, context):
    u_hat, g_hat = u0
    def func():
        return solver.integrate(u_hat, g_hat, rhs, params.dt, solver, context)
    return func

~~~

Another example is `KMMRK3.py` (builds upon `KMM.py`) :

~~~python
from .KMM import *

def get_context():
	(same tasks as in KMM.py) ...
    # RK 3 requires three solvers because of the three different coefficients ...
    hdf5file = KMMRK3Writer({"U":U[0], "V":U[1], "W":U[2]}, ...
    return config.AttributeDict(locals())

class KMMRK3Writer(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)    # updates U from U_hat

@optimizer
def add_linear(rhs, u, g, work, AB, AC, SBB, ABB, BBB, nu, dt, K2, K4, a, b):
     ...
     return rhs

def ComputeRHS(rhs, u_hat, g_hat, rk, solver,
               H_hat, VFSp, FSTp, FSBp, FCTp, work, Kx, K2, K4, hv,
               hg, a, b, la, mat, **context):
    """Compute right hand side of Navier Stokes

    args:
        rhs         The right hand side to be returned
        u_hat       The FST of the velocity at current time.
        g_hat       The FST of the curl in wall normal direction
        rk          The step in the Runge Kutta integrator
        solver      The current solver module

    Remaining args are extracted from context
    """
    # Nonlinear convection term at current u_hat
	...
    rhs = solver.add_linear(rhs, u_hat[0], g_hat, work, mat.AB[rk], mat.AC[rk],
                            mat.SBB, mat.ABB, mat.BBB, params.nu, params.dt,
                            K2, K4, a[rk], b[rk])
    return rhs

def solve_linear(u_hat, g_hat, rhs, rk,
                 work, la, mat, H_hat, Sk, h1, a, b, K_over_K2, **context):
 	...
    return u_hat, g_hat

def integrate(u_hat, g_hat, rhs, dt, solver, context):
    """Three stage Runge Kutta integrator for KMM channel solver"""
    for rk in range(3):
        rhs = solver.ComputeRHS(rhs, u_hat, g_hat, rk, solver, **context)
        u_hat, g_hat = solver.solve_linear(u_hat, g_hat, rhs, rk, **context)
    return (u_hat, g_hat), dt, dt
~~~

## `solvers/spectralinit.py`

Contains essentials like MPI configuration variables (and dummy methods):

~~~python
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
params = config.params
profiler = cProfile.Profile()

def regression_test(context):
def update(context):
def additional_callback(context):
def solve_linear(context):
def conv(*args):
def set_source(Source, **context):
def end_of_tstep(context):
~~~




