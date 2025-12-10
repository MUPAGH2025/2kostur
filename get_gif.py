from Program import *



bathymetry = bathymetrys(grid := (60,60))

dt_over_dxy=(.2, .2)

run0 = WaveByBathymetry(grid, bathymetry.flat(), dt_over_dxy)
run1 = WaveByBathymetry(grid, bathymetry.gauss_(-0.15), dt_over_dxy)
run2 = WaveByBathymetry(grid, bathymetry.gauss_(+0.15), dt_over_dxy)
run3 = WaveByBathymetry(grid, bathymetry.gauss_(-0.30), dt_over_dxy)
run4 = WaveByBathymetry(grid, bathymetry.gauss_(+0.30), dt_over_dxy)

run0.save_anim("out0.gif")
run1.save_anim("out1.gif")
run2.save_anim("out2.gif")
run3.save_anim("out3.gif")
run4.save_anim("out4.gif")


plot_var_gif(run0.get_vh()/run0.get_h(), "vh.gif")