# %%
import xarray as xr
import numpy as np
import src.plots.utils as utils

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import proplot as pplt
from matplotlib.lines import Line2D

from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches


import src.plots.composite_plot as composite_plot
import src.plots.extreme_plot as extplt
import src.plots.statistical_overview as stat_overview
import src.obs.era5_extreme_change as era5_extreme_change
import src.extreme.extreme_count_troposphere as ext_profile

#%%
def read_composite(
    model, var_name, fixed_pattern="decade_mpi", reduction="mean_same_number"
):
    """read composite data"""
    odir = "/work/mh0033/m300883/Tel_MMLE/data/"
    comp_name = f"plev_50000_{fixed_pattern}_first_JJA_JJA_first_last_{var_name}_composite_{reduction}.nc"
    composite = xr.open_dataset(f"{odir}{model}/composite/{comp_name}")
    if var_name == "ts":
        try:
            composite = composite.tsurf
        except AttributeError:
            composite = composite.ts
    elif var_name == "pr":
        try:
            composite = composite.pr
        except AttributeError:
            composite = composite.precip
    elif var_name == 'psl':
        try:
            composite = composite.psl
        except AttributeError:
            composite = composite.slp
    return composite


# %%
models = ["MPI_GE", "CanESM2", "CESM1_CAM5", "MK36", "GFDL_CM3"]

COMPOSITEs_ts = {
    model: read_composite(model, var_name="ts", reduction='mean') for model in models
}

COMPOSITEs_psl = {
    model: read_composite(model, var_name="psl", reduction='mean') for model in models
}

#%%
CR20_composite_ts_first = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/first_composite_mean_ts_40.nc")
CR20_composite_ts_last = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/last_composite_mean_ts_40.nc")
CR20_composite_ts_diff = CR20_composite_ts_last - CR20_composite_ts_first
CR20_composite_ts = xr.concat([CR20_composite_ts_first, CR20_composite_ts_last, CR20_composite_ts_diff], dim="period")
CR20_composite_ts = CR20_composite_ts.ts
CR20_composite_ts['period'] = ["first", "last", "diff"]
#%%
# add to the dictionary
COMPOSITEs_ts["20CR"] = CR20_composite_ts

#%%
CR20_composite_psl_first = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/first_composite_mean_psl_40.nc")
CR20_composite_psl_last = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/last_composite_mean_psl_40.nc")
CR20_composite_psl_diff = CR20_composite_psl_last - CR20_composite_psl_first
CR20_composite_psl = xr.concat([CR20_composite_psl_first, CR20_composite_psl_last, CR20_composite_psl_diff], dim="period")
CR20_composite_psl = CR20_composite_psl.ts # wrong name 
CR20_composite_psl['period'] = ["first", "last", "diff"]
#%%
# add to the dictionary
COMPOSITEs_psl["20CR"] = CR20_composite_psl

#%%
# change units from Pa to hPa
for model in COMPOSITEs_psl:
    if model != 'GFDL_CM3': # alread in hPa
        COMPOSITEs_psl[model] = COMPOSITEs_psl[model] / 100
#%%
models_plot = ["MPI_GE", "CanESM2", "CESM1_CAM5", "MK36", "GFDL_CM3", "20CR"]
models_legend = [
    "MPI-GE (100)",
    "CanESM2 (50)",
    "CESM1-CAM5 (40)",
    "MK3.6 (30)",
    "GFDL-CM3 (20)",
    "20CR_ens (80)",
]



temp_cmap_seq = np.loadtxt(
    "/work/mh0033/m300883/High_frequecy_flow/data/colormaps-master/continuous_colormaps_rgb_0-1/temp_seq.txt"
)
temp_cmap_seq = mcolors.ListedColormap(temp_cmap_seq, name="prec_div")

temp_cmap_div = np.loadtxt(
    "/work/mh0033/m300883/High_frequecy_flow/data/colormaps-master/continuous_colormaps_rgb_0-1/temp_div.txt"
)
temp_cmap_div = mcolors.ListedColormap(temp_cmap_div, name="prec_div")



# %%
def plot_composite_single_ext_rc(COMPOSITEs, models, axes, extr_type="pos", fill = True, **kwargs):
    levels_shading = kwargs.get("levels_shading", np.arange(-1.5, 1.6, 0.3))
    levels_lines = kwargs.get("levels_lines", np.arange(-2, 2.1, 0.5))
    for i, model in enumerate(models):  # rows for different models
        first = COMPOSITEs[model].sel(mode="NAO", period="first", extr_type=extr_type)
        last = COMPOSITEs[model].sel(mode="NAO", period="last", extr_type=extr_type)
        diff = COMPOSITEs[model].sel(mode="NAO", period="diff", extr_type=extr_type)

        try:
            first = utils.erase_white_line(first)
            last = utils.erase_white_line(last)
            diff = utils.erase_white_line(diff)
        except ValueError:
            pass

        data_all = [
            first,
            last,
            diff,
        ]

        maps = []
        for j, data in enumerate(data_all):  # cols for different data
            if fill is True:
                map = axes[i, j].contourf(
                    data,
                    x="lon",
                    y="lat",
                    levels=levels_shading,
                    extend="both",
                    transform=ccrs.PlateCarree(),
                    cmap = temp_cmap_div,
                )
            else:
                map = axes[i, j].contour(
                    data,
                    x="lon",
                    y="lat",
                    levels=[level for level in levels_lines if level != 0],
                    colors = 'k',
                    linewidths = 0.5,
                    extend="both",
                    transform=ccrs.PlateCarree(),
                    zorder = 10,
                )
            maps.append(map)
            axes[i, j].grid(color="grey7", linewidth=0.5)

        # significant area as hatches.
        if fill is True:
            if i < len(models) - 1:
                diff_sig = COMPOSITEs[model].sel(
                    mode="NAO", period="diff_sig", extr_type=extr_type
                )

                diff_sig = utils.erase_white_line(diff_sig)

                axes[i, 2].contourf(
                    diff_sig,
                    levels=[-0.5, 0.5, 1.5],
                    colors=["none", "none"],
                    hatches=["", "xxxxx"],
                    zorder=20,
                )

    return axes, maps

# %%

fig3, axes = pplt.subplots(
    space=0,
    width=180 / 25.4,
    wspace=1.5,
    hspace=0.2,
    proj="ortho",
    proj_kw=({"lon_0": -20, "lat_0": 60}),
    nrows=6,
    ncols=3,
)
axes.format(
    abc = False,
    latlines=20,
    lonlines=30,
    color="grey7",
    coast=True,
    coastlinewidth=0.3,
    coastcolor="charcoal",
    leftlabels=models_legend,
    toplabels=["first", "last", "last - first"],
    toplabels_kw={"fontsize": 10, },
    leftlabels_kw={"fontsize": 10,},
)


axes, maps = plot_composite_single_ext_rc(COMPOSITEs_ts, models_plot, axes)
axes, lines = plot_composite_single_ext_rc(COMPOSITEs_psl, models_plot, axes, fill=False, levels_lines=np.arange(-5, 5.1, 1))

fig3.colorbar(
    maps[0],
    loc="b",
    pad=1,
    title="(near) surface temperature [K]",
    width=0.1,
    shrink=1,
)
# label a, b, c vertically order
axes[0, 0].text(0.1, 1, "a", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
axes[1, 0].text(0.1, 1, "b", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
axes[2, 0].text(0.1, 1, "c", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
axes[3, 0].text(0.1, 1, "d", transform=axes[3, 0].transAxes, fontsize=12, fontweight='bold')
axes[4, 0].text(0.1, 1, "e", transform=axes[4, 0].transAxes, fontsize=12, fontweight='bold')
axes[5, 0].text(0.1, 1, "f", transform=axes[5, 0].transAxes, fontsize=12, fontweight='bold')

axes[0, 1].text(0.1, 1, "g", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold')
axes[1, 1].text(0.1, 1, "h", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
axes[2, 1].text(0.1, 1, "i", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold')
axes[3, 1].text(0.1, 1, "j", transform=axes[3, 1].transAxes, fontsize=12, fontweight='bold')
axes[4, 1].text(0.1, 1, "k", transform=axes[4, 1].transAxes, fontsize=12, fontweight='bold')
axes[5, 1].text(0.1, 1, "l", transform=axes[5, 1].transAxes, fontsize=12, fontweight='bold')

axes[0, 2].text(0.1, 1, "m", transform=axes[0, 2].transAxes, fontsize=12, fontweight='bold')
axes[1, 2].text(0.1, 1, "n", transform=axes[1, 2].transAxes, fontsize=12, fontweight='bold')
axes[2, 2].text(0.1, 1, "o", transform=axes[2, 2].transAxes, fontsize=12, fontweight='bold')
axes[3, 2].text(0.1, 1, "p", transform=axes[3, 2].transAxes, fontsize=12, fontweight='bold')
axes[4, 2].text(0.1, 1, "q", transform=axes[4, 2].transAxes, fontsize=12, fontweight='bold')
axes[5, 2].text(0.1, 1, "r", transform=axes[5, 2].transAxes, fontsize=12, fontweight='bold')

plt.savefig("/work/mh0033/m300883/Tel_MMLE/docs/source/plots/paper_main/ts_composite_pos_rc.pdf", dpi=300, bbox_inches="tight")

# %%
fig4, axes = pplt.subplots(
    space=0,
    abc=False,
    abcloc='ul',  # upper left
    wspace=1.5,
    hspace=0.2,
    proj="ortho",
    proj_kw=({"lon_0": -20, "lat_0": 60}),
    nrows=6,
    ncols=3,
)
axes.format(
    latlines=20,
    lonlines=30,
    color="grey7",
    coast=True,
    coastlinewidth=0.3,
    coastcolor="charcoal",
    leftlabels=models_legend,
    toplabels=["first", "last", "last - first"],
    toplabels_kw={"fontsize": 10,},
    leftlabels_kw={"fontsize": 10,},
)

axes, maps = plot_composite_single_ext_rc(COMPOSITEs_ts, models_plot, axes, 'neg')
axes, lines = plot_composite_single_ext_rc(COMPOSITEs_psl, models_plot, axes, 'neg', fill=False, levels_lines=np.arange(-5, 5.1, 1))
fig4.colorbar(
    maps[0],
    loc="b",
    pad=1,
    title="(near) surface temperature [K]",
    width=0.1,
    shrink=1,
)
# label a, b, c vertically order
axes[0, 0].text(0.1, 1, "a", transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
axes[1, 0].text(0.1, 1, "b", transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
axes[2, 0].text(0.1, 1, "c", transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')
axes[3, 0].text(0.1, 1, "d", transform=axes[3, 0].transAxes, fontsize=12, fontweight='bold')
axes[4, 0].text(0.1, 1, "e", transform=axes[4, 0].transAxes, fontsize=12, fontweight='bold')
axes[5, 0].text(0.1, 1, "f", transform=axes[5, 0].transAxes, fontsize=12, fontweight='bold')

axes[0, 1].text(0.1, 1, "g", transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold')
axes[1, 1].text(0.1, 1, "h", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
axes[2, 1].text(0.1, 1, "i", transform=axes[2, 1].transAxes, fontsize=12, fontweight='bold')
axes[3, 1].text(0.1, 1, "j", transform=axes[3, 1].transAxes, fontsize=12, fontweight='bold')
axes[4, 1].text(0.1, 1, "k", transform=axes[4, 1].transAxes, fontsize=12, fontweight='bold')
axes[5, 1].text(0.1, 1, "l", transform=axes[5, 1].transAxes, fontsize=12, fontweight='bold')

axes[0, 2].text(0.1, 1, "m", transform=axes[0, 2].transAxes, fontsize=12, fontweight='bold')
axes[1, 2].text(0.1, 1, "n", transform=axes[1, 2].transAxes, fontsize=12, fontweight='bold')
axes[2, 2].text(0.1, 1, "o", transform=axes[2, 2].transAxes, fontsize=12, fontweight='bold')
axes[3, 2].text(0.1, 1, "p", transform=axes[3, 2].transAxes, fontsize=12, fontweight='bold')
axes[4, 2].text(0.1, 1, "q", transform=axes[4, 2].transAxes, fontsize=12, fontweight='bold')
axes[5, 2].text(0.1, 1, "r", transform=axes[5, 2].transAxes, fontsize=12, fontweight='bold')

plt.savefig("/work/mh0033/m300883/Tel_MMLE/docs/source/plots/paper_main/ts_composite_neg_rc.pdf", dpi=300, bbox_inches="tight")
# %%
fig, axes = plt.subplots(
    1, 2, figsize = (7, 4),
    subplot_kw={"projection": ccrs.Orthographic(0, 70)},)


map = erase_white_line(COMPOSITEs_ts['MPI_GE'].sel(mode = 'NAO', extr_type = 'pos', period = 'last')).plot.contourf(
    ax=axes[0],
    x="lon",
    y="lat",
    levels=np.arange(-1.5, 1.6, 0.1),
    extend="both",
    transform=ccrs.PlateCarree(),
    cmap=temp_cmap_div,
    add_colorbar=False,
)

# negative
erase_white_line(COMPOSITEs_ts['MPI_GE'].sel(mode = 'NAO', extr_type = 'neg', period = 'last')).plot.contourf(
    ax=axes[1],
    x="lon",
    y="lat",
    levels=np.arange(-1.2, 1.3, 0.1),
    extend="both",
    transform=ccrs.PlateCarree(),
    cmap=temp_cmap_div,
    add_colorbar=False,
)



# add coastlines
axes[0].coastlines(color='black', linewidth=0.5)
axes[1].coastlines(color='black', linewidth=0.5)

# add gridlines
axes[0].gridlines(draw_labels=False, color='gray', linestyle='--', linewidth=0.5)
axes[1].gridlines(draw_labels=False, color='gray', linestyle='--', linewidth=0.5)

# add colorbar vertical at the right side of the second plot
cax = fig.add_axes([0.99, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(map, cax=cax, orientation='vertical')
cbar.set_label('Difference from average temperature [K]', fontsize=11)

axes[0].set_title('positive NAO extreme', fontsize=12)
axes[1].set_title('negative NAO extreme', fontsize=12)

plt.tight_layout()

plt.savefig("/work/mh0033/m300883/Tel_MMLE/docs/source/plots/workshop/ts_composite_pos_neg_mpi_ge.png", dpi=300, bbox_inches="tight")

# %%
