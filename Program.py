import numpy as np
from matplotlib import pyplot
from open_atmos_jupyter_utils import show_anim
from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options
from PyMPDATA.boundary_conditions import Constant
import pandas as pd
import scipy


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

    def gaussian_2d(self, center: tuple, a=-.15):
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

    def gauss_(self, n_p):
        tmp = np.tile(
            np.zeros(self.y_len),
            (self.x_len,1)
        )
        tmp += self.gaussian_2d(center=( self.x_len//6,   self.y_len),a=(-1)**n_p *0.15 )
        tmp += self.gaussian_2d(center=( 5*self.x_len//6, self.y_len),a=(-1)**n_p *0.15 )
        
        return tmp
    

    def liniar(self, n_p):
        tmp = np.tile(
            np.zeros(self.x_len),
            (self.y_len,1)
        )
        tmp[
            5*self.x_len // 6 - self.x_len // 6:
            5*self.x_len // 6 + self.x_len // 6,
            5*self.y_len // 6 - self.y_len // 6:
            5*self.y_len // 6 + self.y_len // 6
        ] = np.tile(
            np.linspace(0, (-1)**n_p * 0.15, self.y_len // 3 ),
            (self.x_len // 3, 1)
        )
        tmp[
            self.x_len // 6 - self.x_len // 6:
            self.x_len // 6 + self.x_len // 6,
            5*self.y_len // 6 - self.y_len // 6:
            5*self.y_len // 6 + self.y_len // 6
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

        self.pillars_l = grid[0]//6.

        self.Force_arr = []

        #  Przeliczenie siatni na metry
        self.dx = 10/self.x_len
        self.dy = 10/self.y_len
        self.dt = 0.25
        self.dz = 1

        #  Fala na początku ?
        h_initial = .5 * np.ones(grid) + bathymetry.copy()
        h_initial[
            grid[0] // 2 - grid[0] // 2:
            grid[0] // 2 + grid[0] // 2,
            grid[1] // 5 - grid[1] // 18:
            grid[1] // 5 + grid[1] // 18
        ] += .025

        self.outfreq = int(self.y_len//20)
        self.output = ShallowWaterEquationsIntegrator(
            h_initial=h_initial,
            options=Options(nonoscillatory=True, infinite_gauge=True),
            bathymetry=bathymetry
        )(
            nt=int(1.6*self.y_len)+30, g=scipy.constants.g, dt_over_dxy=dt_over_dxy, outfreq=(self.outfreq)
        )

    def get_h_last_frame(self):
        # return  self.output['h'][-1]-self.bathymetry
        return  self.output['h']-self.bathymetry
    
    def get_uh_last_frame(self):
        return  self.output['uh'][-1]
    
    def get_vh_last_frame(self):
        # return  self.output['vh'][-1]
        return  self.output['vh']
    
    #  Do poprawy
    def calc_force(self, frame):
        rho_w = 997.

        x_range = slice(0, self.x_len // 3)
        y_range = slice(5*self.y_len//6,self.y_len )

        dV =  (self.output['h'][frame][x_range, y_range] -self.bathymetry[x_range, y_range]-0.5) * self.x_len // 3 * self.y_len//6
        

        dv_dt = (self.output['vh'][frame][
            x_range, y_range
        ] - self.output['vh'][frame-1] [
            x_range, y_range 
        ])/self.dt
                
        return np.abs( np.sum(dV *dv_dt) )*rho_w

    def get_cube(self):   
        phi = np.arange(1,10,2)*np.pi/4
        Phi, Theta = np.meshgrid(phi, phi) 
        x = np.cos(Phi)*np.sin(Theta)
        y = np.sin(Phi)*np.sin(Theta)
        z = np.cos(Theta)/np.sqrt(2)
        return x,y,z

    def plot(self, frame, *, zlim=(.45, .55)):
        filar_cord = pd.DataFrame({"x": [5*self.pillars_l, self.pillars_l], "y" : [13*self.y_len/12., 13*self.y_len/12.], "z" : [0.5, 0.5]})
        
        psi = self.output['h'][frame]-self.bathymetry
        xi, yi = np.indices(psi.shape)
        # zlim *= self.dz

        Force = self.calc_force(frame) if frame !=1 else 0
        self.Force_arr.append(Force)

        fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 6))

        #  Draw pillars
        for i in filar_cord.index:
            x,y,z = self.get_cube()
            x = (x*self.pillars_l + filar_cord.x[i])*self.dx
            y = (y*self.pillars_l + filar_cord.y[i])*self.dy
            z = (z*.1 + filar_cord.z[i])*self.dz
            ax.plot_surface(x, y, z, color="Black")

        #  Draw wave
        ax.plot_wireframe((xi+.5)*self.dx, (yi+.5)*self.dy, psi, color='blue', linewidth=.5)
        ax.set(zlim=zlim, proj_type='ortho', title=f"t = {frame * self.dt:.2E} [s], F = {Force:.2E} [N]", zlabel="$\zeta$")

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

