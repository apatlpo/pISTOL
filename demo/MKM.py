"""Turbulent channel"""
import warnings
from mpi4py import MPI
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cbook
from mpiFFT4py import dct
#from spectralDNS.utilities import reset_profile
from spectralDNS import config, get_solver, solve

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Use constant flux and adjust pressure gradient dynamically
#flux = np.array([1645.46])
flux = np.array([736.43])

def initOS(OS, U, X, t=0.):
    for i in range(U.shape[1]):
        x = X[0][i, 0, 0]
        OS.interp(x)
        for j in range(U.shape[2]):
            y = X[1][i, j, 0]
            v = np.dot(OS.f, np.real(OS.dphidy*np.exp(1j*(y-OS.eigval*t))))
            u = -np.dot(OS.f, np.real(1j*OS.phi*np.exp(1j*(y-OS.eigval*t))))
            U[0, i, j, :] = u
            U[1, i, j, :] = v
    U[2] = 0

def initialize(solver, context):
    # Initialize with pertubation ala perturbU (https://github.com/wyldckat/perturbU) for openfoam
    U = context.U
    X = context.X
    params = config.params
    Y = np.where(X[0] < 0, 1+X[0], 1-X[0])
    utau = params.nu*params.Re_tau
    #Um = 46.9091*utau
    Um = 56.*utau
    Xplus = Y*params.Re_tau
    Yplus = X[1]*params.Re_tau
    Zplus = X[2]*params.Re_tau
    duplus = Um*0.2/utau  #Um*0.25/utau
    alfaplus = params.L[1]/500.
    betaplus = params.L[2]/200.
    sigma = 0.00055 # 0.00055
    epsilon = Um/200.   #Um/200.
    U[:] = 0
    U[1] = Um*(Y-0.5*Y**2)
    dev = 1+0.0001*np.random.randn(Y.shape[0], Y.shape[1], Y.shape[2])
    #dev = np.fromfile('dev.dat').reshape((64, 64, 64))
    dd = utau*duplus/2.0*Xplus/40.*np.exp(-sigma*Xplus**2+0.5)*np.cos(betaplus*Zplus)*dev
    U[1] += dd
    U[2] += epsilon*np.sin(alfaplus*Yplus)*Xplus*np.exp(-sigma*Xplus**2)*dev

    U_hat = solver.set_velocity(**context)
    U = solver.get_velocity(**context)
    U_hat = solver.set_velocity(**context)

    if "KMM" in params.solver:
        context.g[:] = 1j*context.K[1]*U_hat[2] - 1j*context.K[2]*U_hat[1]

    # Set the flux
    #flux[0] = context.FST.dx(U[1], context.ST.quad)
    #solver.comm.Bcast(flux)

    if solver.rank == 0:
        print("Flux {}".format(flux[0]))

    if not 'KMM' in params.solver:
        P_hat = solver.compute_pressure(**context)
        P = context.FST.backward(P_hat, context.P, context.SN)

    if not 'RK3' in params.solver:
        context.U_hat0[:] = context.U_hat[:]
        context.H_hat1[:] = solver.get_convection(**context)

def init_from_file(filename, solver, context):
    f = h5py.File(filename, driver="mpio", comm=solver.comm)
    assert "1" in f["3D/checkpoint/U"]
    U = context.U
    N = U.shape[1]
    s = slice(solver.rank*N, (solver.rank+1)*N, 1)

    # previous timestep
    if not 'RK3' in config.params.solver:
        assert "0" in f["3D/checkpoint/U"]

        U[:] = f["3D/checkpoint/U/0"][:, s]
        U_hat = solver.set_velocity(**context)

        # Set g, which is used in computing convection
        if "KMM" in config.params.solver:
            context.g[:] = 1j*context.K[1]*U_hat[2] - 1j*context.K[2]*U_hat[1]

        context.U_hat0[:] = U_hat
        context.H_hat1[:] = solver.get_convection(**context)

    # current timestep
    U[:] = f["3D/checkpoint/U/1"][:, s]
    U_hat = solver.set_velocity(**context)

    if config.params.solver in ("IPCS", "IPCSR"):
        context.P[:] = f["3D/checkpoint/P/1"][s]
        solver.set_pressure(**context)

    elif "KMM" in config.params.solver:
        context.g[:] = 1j*context.K[1]*U_hat[2] - 1j*context.K[2]*U_hat[1]

    f.close()

def set_Source(Source, Sk, ST, FST, **context):
    utau = config.params.nu * config.params.Re_tau
    Source[:] = 0
    Source[1, :] = -utau**2
    Sk[:] = 0
    if hasattr(FST, 'complex_shape'):
        Sk[1] = FST.scalar_product(Source[1], Sk[1], ST)
    else:
        Sk[1] = FST.scalar_product(Source[1], Sk[1])

def dx(u, FST):
    """Compute integral of u over domain"""
    uu = np.sum(u, axis=(1, 2))
    sl = FST.local_slice(False)[0]
    M = FST.shape()[0]
    c = np.zeros(M)
    cc = np.zeros(M)
    cc[sl] = uu
    FST.comm.Reduce(cc, c, op=MPI.SUM, root=0)
    quad = FST.bases[0].quad
    if FST.comm.Get_rank() == 0:
        if quad == 'GL':
            ak = np.zeros_like(c)
            ak = dct(c, ak, 1, axis=0)
            ak /= (M-1)
            w = np.arange(0, M, 1, dtype=float)
            w[2:] = 2./(1-w[2:]**2)
            w[0] = 1
            w[1::2] = 0
            return sum(ak*w)*config.params.L[1]*config.params.L[2]/config.params.N[1]/config.params.N[2]

        assert quad == 'GC'
        d = np.zeros(M)
        k = 2*(1 + np.arange((M-1)//2))
        d[::2] = (2./M)/np.hstack((1., 1.-k*k))
        w = np.zeros_like(d)
        w = dct(d, w, type=3, axis=0)
        return np.sum(c*w)*config.params.L[1]*config.params.L[2]/config.params.N[1]/config.params.N[2]
    return 0

beta = np.zeros(1)
def update(context):
    global im1, im2, im3, flux

    c = context
    params = config.params
    solver = config.solver
    X, U, U_hat = c.X, c.U, c.U_hat

    #if params.tstep == 1: reset_profile(profile)

    # Dynamically adjust flux
    if params.tstep % 1 == 0:
        if hasattr(c.FST, 'complex_shape'):
            U[1] = c.FST.backward(U_hat[1], U[1], c.ST)
            beta[0] = c.FST.dx(U[1], c.ST.quad)
        else:
            U[1] = c.FST.backward(U_hat[1], U[1])
            beta[0] = dx(U[1], c.FST)

        #solver.comm.Bcast(beta)
        q = (flux[0] - beta[0])  # array(params.L).prod()
        #U_tmp = c.work[(U[0], 0)]
        #F_tmp = c.work[(U_hat[0], 0)]
        #U_tmp[:] = beta[0]
        #F_tmp = c.FST.forward(U_tmp, F_tmp, c.ST)
        #U_hat[1] += q/beta[0]*U_hat[1]
        if solver.rank == 0:
            #d0 = zeros(U_hat.shape[1], dtype=U_hat.dtype)
            #d1 = zeros(U_hat.shape[1], dtype=U_hat.dtype)
            #d0 = c.mat.ADD.matvec(U_hat[1,:,0,0], d0)
            #d1 = c.mat.BDD.matvec(U_hat[1,:,0,0], d1)
            #c.Sk[1,0,0,0] -= (flux[0]/beta[0]-1)/params.dt*(-params.nu*params.dt/2.*d0[0] + d1[0])*0.025
            c.Sk[1, 0, 0, 0] -= (flux[0]/beta[0]-1)*0.1

        #c.Source[1] -= q/array(params.L).prod()
        #c.Sk[1] = c.FST.scalar_product(c.Source[1], c.Sk[1], c.ST)

    #if params.tstep % 1 == 0:
        #U[1] = c.FST.backward(U_hat[1], U[1], c.ST)
        #beta[0] = c.FST.dx(U[1], c.ST.quad)
        ##beta[0] = (flux[0] - beta[0])/(array(params.L).prod())
        #solver.comm.Bcast(beta)
        #q = flux[0]/beta[0]-1
        ##U[1] += beta[0]*U[1]
        ##U_hat[1] = c.FST.forward(U[1], U_hat[1], c.ST)
        #U_hat[1] += q*U_hat[1]
        #c.Source[1] -= beta[0]*q/(array(params.L).prod())/params.dt/2
        #c.Sk[1] = c.FST.scalar_product(c.Source[1], c.Sk[1], c.ST)

    #utau = config.params.Re_tau * config.params.nu
    #Source[:] = 0
    #Source[1] = -utau**2
    #Source[:] += 0.05*random.randn(*U.shape)
    #for i in range(3):
        #Sk[i] = FST.scalar_product(Source[i], Sk[i], ST)

    if (params.tstep % params.compute_energy == 0 or
            params.tstep % params.plot_result == 0 and params.plot_result > 0 or
            params.tstep % params.sample_stats == 0):
        U = solver.get_velocity(**c)

    if params.tstep % params.print_energy0 == 0 and solver.rank == 0:
        print((c.U_hat[0].real*c.U_hat[0].real).mean(axis=(0, 2)))
        print((c.U_hat[0].real*c.U_hat[0].real).mean(axis=(0, 1)))
        #print(params.tstep, (c.U_hat[0]*c.U_hat[0]).mean()/4096**2, (c.U[0]*c.U[0]).mean())
        #print(params.tstep, (c.U_hat[1]*c.U_hat[1]).mean()/4096**2, (c.U[1]*c.U[1]).mean())
        #print(params.tstep, (c.U_hat[2]*c.U_hat[2]).mean()/4096**2, (c.U[2]*c.U[2]).mean())

    if params.tstep == 1 and solver.rank == 0 and params.plot_result > 0:
        # Initialize figures
        plt.figure()
        im1 = plt.contourf(X[1][:, :, 0], X[0][:, :, 0], U[0, :, :, 0], 100)
        plt.colorbar(im1)
        plt.draw()

        plt.figure()
        im2 = plt.contourf(X[1][:, :, 0], X[0][:, :, 0], U[1, :, :, 0], 100)
        plt.colorbar(im2)
        plt.draw()

        plt.figure()
        im3 = plt.contourf(X[2][:, 0, :], X[0][:, 0, :], U[0, :, 0, :], 100)
        plt.colorbar(im3)
        plt.draw()

        plt.pause(1e-6)

    if params.tstep % params.plot_result == 0 and solver.rank == 0 and params.plot_result > 0:
        im1.ax.clear()
        im1.ax.contourf(X[1][:, :, 0], X[0][:, :, 0], U[0, :, :, 0], 100)
        im1.autoscale()
        plt.figure(1)
        plt.pause(1e-6)
        im2.ax.clear()
        im2.ax.contourf(X[1][:, :, 0], X[0][:, :, 0], U[1, :, :, 0], 100)
        im2.autoscale()
        plt.figure(2)
        plt.pause(1e-6)
        im3.ax.clear()
        #im3.ax.contourf(X[1, :,:,0], X[0, :,:,0], P[:, :, 0], 100)
        im3.ax.contourf(X[2][:, 0, :], X[0][:, 0, :], U[0, :, 0, :], 100)
        im3.autoscale()
        plt.figure(3)
        plt.pause(1e-6)

    if params.tstep % params.compute_energy == 0:
        if hasattr(c.FST, 'complex_shape'):
            e0 = c.FST.dx(U[0]*U[0], c.ST.quad)
            e1 = c.FST.dx(U[1]*U[1], c.ST.quad)
            e2 = c.FST.dx(U[2]*U[2], c.ST.quad)
            q = c.FST.dx(U[1], c.ST.quad)
        else:
            e0 = dx(U[0]*U[0], c.FST)
            e1 = dx(U[1]*U[1], c.FST)
            e2 = dx(U[2]*U[2], c.FST)
            q = dx(U[1], c.FST)
        if solver.rank == 0:
            print("Time %2.5f Energy %2.8e %2.8e %2.8e Flux %2.6e Q %2.6e %2.6e %2.6e" %(config.params.t, e0, e1, e2, q, e0+e1+e2, c.Sk[1, 0, 0, 0], flux[0]/beta[0]-1))

    if params.tstep % params.sample_stats == 0:
        solver.stats(U)

    #if params.tstep == 2:
        #solver.fastest_timestep = 1e8
        #solver.slowest_timestep = 0

    #if params.tstep == 1:
        #print "Reset profile"
        #reset_profile(profile)

class Stats(object):

    def __init__(self, U, comm, fromstats="", filename=""):
        self.shape = U.shape[1:]
        self.Umean = np.zeros(U.shape[:2])
        self.Pmean = np.zeros(U.shape[1])
        self.UU = np.zeros((6, U.shape[1]))
        self.num_samples = 0
        self.rank = comm.Get_rank()
        self.fname = filename
        self.comm = comm
        self.f0 = None
        if fromstats:
            self.fromfile(filename=fromstats)

    def create_statsfile(self):
        self.f0 = h5py.File(self.fname+".h5", "w", driver="mpio", comm=self.comm)
        self.f0.create_group("Average")
        self.f0.create_group("Reynolds Stress")
        for i in ("U", "V", "W", "P"):
            self.f0["Average"].create_dataset(i, shape=(2**config.params.M[0],), dtype=float)

        for i in ("UU", "VV", "WW", "UV", "UW", "VW"):
            self.f0["Reynolds Stress"].create_dataset(i, shape=(2**config.params.M[0], ), dtype=float)

    def __call__(self, U, P=None):
        self.num_samples += 1
        self.Umean += sum(U, axis=(2, 3))
        if not P is None:
            self.Pmean += sum(P, axis=(1, 2))
        self.UU[0] += sum(U[0]*U[0], axis=(1, 2))
        self.UU[1] += sum(U[1]*U[1], axis=(1, 2))
        self.UU[2] += sum(U[2]*U[2], axis=(1, 2))
        self.UU[3] += sum(U[0]*U[1], axis=(1, 2))
        self.UU[4] += sum(U[0]*U[2], axis=(1, 2))
        self.UU[5] += sum(U[1]*U[2], axis=(1, 2))
        self.get_stats()

    def get_stats(self, tofile=True):
        N = self.shape[0]
        s = slice(self.rank*N, (self.rank+1)*N, 1)
        Nd = self.num_samples*self.shape[1]*self.shape[2]
        self.comm.barrier()
        if tofile:
            if self.f0 is None:
                self.create_statsfile()
            else:
                self.f0 = h5py.File(self.fname+".h5", "a", driver="mpio", comm=self.comm)

            for i, name in enumerate(("U", "V", "W")):
                self.f0["Average/"+name][s] = self.Umean[i]/Nd
            self.f0["Average/P"][s] = self.Pmean/Nd
            for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
                self.f0["Reynolds Stress/"+name][s] = self.UU[i]/Nd
            self.f0.attrs.create("num_samples", self.num_samples)
            self.f0.close()

        if self.comm.Get_size() == 1:
            return self.Umean/Nd, self.Pmean/Nd, self.UU/Nd
        return 0

    def reset_stats(self):
        self.num_samples = 0
        self.Umean[:] = 0
        self.Pmean[:] = 0
        self.UU[:] = 0

    def fromfile(self, filename="stats"):
        self.fname = filename
        self.f0 = h5py.File(filename+".h5", "a", driver="mpio", comm=self.comm)
        N = self.shape[0]
        self.num_samples = self.f0.attrs["num_samples"]
        Nd = self.num_samples*self.shape[1]*self.shape[2]
        s = slice(self.rank*N, (self.rank+1)*N, 1)
        for i, name in enumerate(("U", "V", "W")):
            self.Umean[i, :] = self.f0["Average/"+name][s]*Nd
        self.Pmean[:] = self.f0["Average/P"][s]*Nd
        for i, name in enumerate(("UU", "VV", "WW", "UV", "UW", "VW")):
            self.UU[i, :] = self.f0["Reynolds Stress/"+name][s]*Nd
        self.f0.close()

if __name__ == "__main__":
    config.update(
        {'nu': 1./590.,                  # Viscosity
         'Re_tau': 590.,
         'dt': 0.0005,                  # Time step
         'T': 100.,                    # End time
         'L': [2, 2*np.pi, np.pi],
         'M': [6, 6, 6]
        }, "channel"
    )
    config.channel.add_argument("--compute_energy", type=int, default=10)
    config.channel.add_argument("--plot_result", type=int, default=100)
    config.channel.add_argument("--sample_stats", type=int, default=1000)
    config.channel.add_argument("--print_energy0", type=int, default=10)
    #solver = get_solver(update=update, mesh="channel")
    solver = get_solver(update=update, mesh="channel")
    context = solver.get_context()
    initialize(solver, context)
    #init_from_file("KMM665c.h5", solver, context)
    set_Source(**context)
    solver.stats = Stats(context.U, solver.comm, filename="KMMstatsq")
    context.hdf5file.fname = "KMM665d.h5"
    solve(solver, context)
