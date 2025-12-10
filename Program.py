import numpy as np
from matplotlib import pyplot
import matplotlib.animation as animation
from open_atmos_jupyter_utils import show_anim
from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options
from PyMPDATA.boundary_conditions import Constant
import scipy
import pandas as pd


class ShallowWaterEquationsIntegrator:
    def __init__(self, *, h_initial: np.ndarray, options: Options, bathymetry: np.ndarray):
        X, Y, grid = 0, 1, h_initial.shape
        self.bathymetry = bathymetry
        kwargs = { 
            "boundary_conditions": [Constant(value=0)] * len(grid),
            "halo": options.n_halo,
        }

        self.advector = VectorField((
            np.zeros((grid[X] + 1, grid[Y])),
            np.zeros((grid[X], grid[Y] + 1)) 
        ), **kwargs)

        stepper = Stepper(options=options, grid=grid, n_threads=1)

        self.solvers = { k: Solver(stepper, v, self.advector) for k, v in {
            "h": ScalarField(h_initial, **kwargs),
            "uh": ScalarField(np.zeros(grid), **kwargs),
            "vh": ScalarField(np.zeros(grid), **kwargs),
        }.items() }

    def __getitem__(self, key):
        return self.solvers[key].advectee.get()

    def _apply_half_rhs(self, *, key, axis, g_times_dt_over_dxy):
        self[key][:] -= .5 * g_times_dt_over_dxy * self['h'] * np.gradient(self['h']-self.bathymetry, axis=axis)

    def _update_courant_numbers(self, *, axis, key, mask, dt_over_dxy):
        velocity = np.where(mask, np.nan, 0)
        momentum = self[key]
        np.divide(momentum, self['h'], where=mask, out=velocity)

        whole = slice(None, None) 
        all_but_last = slice(None, -1)
        all_but_first_and_last = slice(1, -1)

        velocity_at_cell_boundaries = velocity[( 
            (all_but_last, whole),
            (whole, all_but_last),
        )[axis]] + np.diff(velocity, axis=axis) / 2 
        courant_number = self.advector.get_component(axis)[(
            (all_but_first_and_last, whole),
            (whole, all_but_first_and_last)
        )[axis]]
        courant_number[:] = velocity_at_cell_boundaries * dt_over_dxy[axis]
        assert np.amax(np.abs(courant_number)) <= 1

    def __call__(self, *, nt: int, g: float, dt_over_dxy: tuple, outfreq: int, eps: float=1e-7):
        output = {k: [] for k in self.solvers.keys()}
        for it in range(nt + 1): 
            if it != 0:
                mask = self['h'] > eps
                for axis, key in enumerate(("uh", "vh")):
                    self._update_courant_numbers(axis=axis, key=key, mask=mask, dt_over_dxy=dt_over_dxy)
                self.solvers["h"].advance(n_steps=1)
                for axis, key in enumerate(("uh", "vh")):
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
                    self.solvers[key].advance(n_steps=1)
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
            if it % outfreq == 0:
                for key in self.solvers.keys():
                    output[key].append(self[key].copy())
        return output

class bathymetrys():

    def __init__(self, grid: tuple):
        self.grid = grid
        self.x_len = grid[0]
        self.y_len = grid[1]

    def gaussian_2d(self, center: tuple, a: int):
        sigma_x=self.y_len/4
        simga_y=self.x_len/7.5

        y = np.arange(self.x_len )
        x = np.arange(self.y_len)
        #  X,Y zamienione miejscami
        xx, yy = np.meshgrid(x, y)
        y0, x0 = center

        z = a * np.exp( -( (xx - x0)**2 / (2 * sigma_x**2) + (yy - y0)**2 / (2 * simga_y**2)  )     )
        return z
    
    def flat(self):
        tmp = np.tile(
            np.zeros(self.y_len),
            (self.x_len,1)
        )
        return tmp

    def gauss_(self, amp: int):
        tmp = np.tile(
            np.zeros(self.y_len),
            (self.x_len,1)
        )
        tmp += self.gaussian_2d(center=( self.x_len//6,   self.y_len),a=amp )
        tmp += self.gaussian_2d(center=( 5*self.x_len//6, self.y_len),a=amp )
        
        return tmp
    

    def liniar(self, n_p):
        tmp = np.tile(
            np.zeros(self.x_len),
            (self.y_len,1)
        )
        tmp[
            5*self.x_len // 6 - self.x_len // 6:
            5*self.x_len // 6 + self.x_len // 6,
            5*self.y_len // 4 - self.y_len // 4:
            5*self.y_len // 4 + self.y_len // 4
        ] = np.tile(
            np.linspace(0, (-1)**n_p * 0.15, self.y_len // 3 ),
            (self.x_len // 3, 1)
        )
        tmp[
            self.x_len // 6 - self.x_len // 6:
            self.x_len // 6 + self.x_len // 6,
            5*self.y_len // 4 - self.y_len // 4:
            5*self.y_len // 4 + self.y_len // 4
        ] = np.tile(
            np.linspace(0,(-1)**n_p *0.15, self.y_len // 3 ),
            (self.x_len // 3, 1)
        )
        return tmp

class WaveByBathymetry:
    
    def __init__(self, grid: np.ndarray, bathymetry: np.ndarray, dt_over_dxy: tuple):
        self.bathymetry = bathymetry

        self.x_len = grid[0]
        self.y_len = grid[1]

        self.pillars_l_x = grid[0]//6.
        self.pillars_l_y = grid[1]//6.

        

        #  Przeliczenie siatni na metry
        self.dx = 10/self.x_len
        self.dy = 10/self.y_len
        self.dt = 0.25
        self.dz = 1

        #  Fala na początku ?
        h_initial = .5 * np.ones(grid) + bathymetry.copy()
        for i in range(0, grid[1]//5 +1):
            h_initial[
                0:grid[0],
                i
            ] += 0.03*np.cos(i*np.pi*5/grid[1]/2)

        self.outfreq = int(self.y_len//20)
        self.output = ShallowWaterEquationsIntegrator(
            h_initial=h_initial,
            options=Options(nonoscillatory=True, infinite_gauge=True),
            bathymetry=bathymetry
        )(
            nt=int(2.5*self.y_len), g=scipy.constants.g, dt_over_dxy=dt_over_dxy, outfreq=(self.outfreq)
        )

        self.Momentum_arr = np.array( [ self.calc_p(i) for i in range(len(self.output['h'])) ] )

    def get_h(self):
        return  self.output['h']-self.bathymetry
    
    def get_uh(self):
        return  self.output['uh']
    
    def get_vh(self):
        return  self.output['vh']
    
    def mom2F(self, frame):
        return (self.Momentum_arr[frame] - self.Momentum_arr[frame-1] ) /self.dt
    
    #  Do poprawy
    def calc_p(self, frame):
        rho_w = 997. #* si.kg / si.m**3
        dH_now =  self.output['h'][frame][:, -1] - .5
        # V_now = np.sum( dH_now[int(self.pillars_l_x//2) : int(3*self.pillars_l_x//2) ] *self.dx)*self.dy
        # Dv_now = self.output['vh'][frame][:, -1]/dH_now

        V_now = np.sum( self.dx)/self.dy
        # V_now = 1 
        Dv_now = self.output['vh'][frame][:, -1]
        dm = V_now*rho_w


        Dv_at_filar = np.sum( Dv_now[int(self.pillars_l_x//2) : int(3*self.pillars_l_x//2) ] )

        return Dv_at_filar * dm

    def get_cube(self):   
        phi = np.arange(1,10,2)*np.pi/4
        Phi, Theta = np.meshgrid(phi, phi) 
        x = np.cos(Phi)*np.sin(Theta)
        y = np.sin(Phi)*np.sin(Theta)
        z = np.cos(Theta)/np.sqrt(2)
        return x,y,z

    def plot(self, frame, *, zlim=(.45, .55)):
        filar_cord = pd.DataFrame({
            "x" : [self.pillars_l_x, self.x_len-self.pillars_l_x],
            "y" : [self.y_len+self.pillars_l_y//2, self.y_len+self.pillars_l_y//2],
            "z" : [0.5, 0.5]
        })
        
        psi = self.output['h'][frame]-self.bathymetry
        xi, yi = np.indices(psi.shape)


        fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 6))

        #  Draw pillars
        for i in filar_cord.index:
            x,y,z = self.get_cube()
            x = (x*self.pillars_l_x + filar_cord.x[i])*self.dx
            y = (y*self.pillars_l_y + filar_cord.y[i])*self.dy
            z = (z*.1 + filar_cord.z[i])*self.dz
            ax.plot_surface(x, y, z, color="Black")

        #  Draw wave
        ax.plot_wireframe((xi+.5)*self.dx, (yi+.5)*self.dy, psi, color='blue', linewidth=.5)
        ax.set(zlim=zlim, proj_type='ortho', title=f"t = {frame * self.dt:.2E} [s]", zlabel="$\zeta$")

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor('black')
            axis.pane.set_alpha(1)
        for axis in ('x', 'y'):
            getattr(ax, f'set_{axis}label')(f"{axis} [m]")

        pyplot.colorbar(
            ax.contourf((xi+.5)*self.dx, (yi+.5)*self.dy, self.bathymetry, zdir='z', offset=zlim[0]),
            pad=.1, aspect=10, fraction=.02, label='bathymetry', location='left'
        ).ax.invert_yaxis()
            
        return fig
        
    def save_anim(self, file_name: str):
        show_anim(self.plot, range(len(self.output['h'])), gif_file=file_name)


def plot_F_p_(run1, str):
    calc_F = [run1.mom2F(i) for i in range(len(run1.Momentum_arr)-1)]
    calc_P = run1.Momentum_arr
    x_range = np.arange(len(run1.output['h']))*run1.dt

    fig, (ax1, ax2) = pyplot.subplots(2, sharex=True)

    ax1.plot(x_range[:-1], calc_F, color='b', label=str)
    ax1.set_ylabel("Siła F [a.u]", color='b')

    ax2.plot(x_range, calc_P, color='g', label=str)
    ax2.set_ylabel("Pęd Fali p [a.u.]", color='g')

    ax1.grid(color='r', linestyle='-', linewidth=0.1)
    ax2.grid(color='r', linestyle='-', linewidth=0.1)

    ax2.set_xlabel("Czas[s]")
    pyplot.show()



def plot_F_4(r1, r2, r3, r4):
    F1 =  [r1.mom2F(i) for i in range(1,len(r1.output['h']))]
    F2 =  [r2.mom2F(i) for i in range(1,len(r2.output['h']))]
    F3 =  [r3.mom2F(i) for i in range(1,len(r3.output['h']))]
    F4 =  [r4.mom2F(i) for i in range(1,len(r4.output['h']))]

    x_range = np.arange(len(r1.output['h']))*r1.dt

    fig, ax = pyplot.subplots(nrows=2, ncols=2, figsize=(12, 8))

    ax[0, 0].plot(x_range[:-1], F1, color='b', label="A=-0.15")
    ax[1, 0].plot(x_range[:-1], F2, color='b', label="A= 0.15")
    ax[0, 1].plot(x_range[:-1], F3, color='b', label="A=-0.30")
    ax[1, 1].plot(x_range[:-1], F4, color='b', label="A= 0.30")
    
    for row in ax:
        for col in row:
            col.grid(color='r', linestyle='-', linewidth=0.1)
            col.set_xlabel("Czas[s]")
            col.set_ylabel("Siła F [a.u]", color='b')
            col.legend()

    pyplot.show()

def plot_P_4(r1, r2, r3, r4):
    F1 =  r1.Momentum_arr
    F2 =  r2.Momentum_arr
    F3 =  r3.Momentum_arr
    F4 =  r4.Momentum_arr

    x_range = np.arange(len(r1.Momentum_arr))*r1.dt

    fig, ax = pyplot.subplots(nrows=2, ncols=2, figsize=(12, 8))

    ax[0, 0].plot(x_range, F1, color='g', label="A=-0.15")
    ax[1, 0].plot(x_range, F2, color='g', label="A= 0.15")
    ax[0, 1].plot(x_range, F3, color='g', label="A=-0.30")
    ax[1, 1].plot(x_range, F4, color='g', label="A= 0.30")
    
    for row in ax:
        for col in row:
            col.grid(color='r', linestyle='-', linewidth=0.1)
            col.set_xlabel("Czas[s]")
            col.set_ylabel("Pęd warstwy fali P [a.u]", color='g')
            col.legend()

    pyplot.show()


def plot_var_gif(psi, file_name):
    fig = pyplot.figure()
    Y, X = np.indices(psi[1].shape)
    ax = pyplot.axes()  
    pyplot.xlabel(r'x')
    pyplot.ylabel(r'y')
    pyplot.colorbar(pyplot.contourf(X, Y, psi[1], cmap ="bone"))
    def animate(i): 
        asd = psi[i]
        cs = pyplot.contourf(X, Y, asd, cmap ="bone") 
        return cs  
    anim = animation.FuncAnimation(fig, animate, frames=len(psi))
    anim.save(file_name)


def plot_var_frame(psi, frame):
    fig = pyplot.figure()
    Y, X = np.indices(psi[1].shape)
    ax = pyplot.axes()  
    pyplot.xlabel(r'x')
    pyplot.ylabel(r'y')
    pyplot.colorbar(pyplot.contourf(X, Y, psi[1], cmap ="bone"))

    pyplot.contourf(X, Y, psi[frame], cmap ="bone") 
