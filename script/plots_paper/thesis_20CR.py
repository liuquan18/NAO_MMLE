
#%%
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import seaborn as sns

from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches


# %%
def read_eof_rean(model, group_size=40):
    odir = "/work/mh0033/m300883/Tel_MMLE/data/" + model + "/"
    first_eof_path = odir + "EOF_result/first_plev500_eof.nc"
    last_eof_path = odir + "EOF_result/last_plev500_eof.nc"

    first_eof = xr.open_dataset(first_eof_path)
    last_eof = xr.open_dataset(last_eof_path)
    return first_eof, last_eof


# read extreme counts
def read_extrc_rean(model, group_size=40):
    odir = f"/work/mh0033/m300883/Tel_MMLE/data/{model}/extreme_count/"
    first_extc_path = odir + "first_plev50000_extc.nc"
    last_extc_path = odir + "last_plev50000_extc.nc"

    first_extc = xr.open_dataset(first_extc_path)
    last_extc = xr.open_dataset(last_extc_path)
    return first_extc, last_extc

def split_first_last(eof_result):
    times = eof_result.time
    years = np.unique(times.dt.year)
    first_years = years[:10]
    last_years = years[-10:]

    eof_first = eof_result.isel(decade=0).sel(
        time=eof_result["time.year"].isin(first_years)
    )
    eof_last = eof_result.isel(decade=-1).sel(
        time=eof_result["time.year"].isin(last_years)
    )
    return eof_first, eof_last

def y_yerr(extrc):
    pos_true = extrc.sel(mode="NAO", confidence="true", extr_type="pos").pc.values
    pos_high = extrc.sel(mode="NAO", confidence="high", extr_type="pos").pc.values
    pos_low = extrc.sel(mode="NAO", confidence="low", extr_type="pos").pc.values
    pos_err = [pos_true - pos_low, pos_high - pos_true]

    neg_true = extrc.sel(mode="NAO", confidence="true", extr_type="neg").pc.values
    neg_high = extrc.sel(mode="NAO", confidence="high", extr_type="neg").pc.values
    neg_low = extrc.sel(mode="NAO", confidence="low", extr_type="neg").pc.values
    neg_err = [neg_true - neg_low, neg_high - neg_true]

    return pos_true, pos_err, neg_true, neg_err

def format_period_year(period, tick_number):
    if period == 0.2:
        # formater = f"first \n 1950-1979"
        formater = "first40"
    elif period == 0.8:
        # formater = f"last \n 1986-2015"
        formater = "last40"
    else:
        formater = None
        print(period)
    return formater


def reananlysis_bar(first_extrc, first_err, last_extrc, last_err, ax, 
                    x = [0.2,0.8],width = 0.4,facecolor = 'none',edgecolor = 'black',linewidth = 1,errcolor = 'black'):

    ax.bar(
        x =x,
        height = [first_extrc,last_extrc],
        width = width,
        edgecolor = edgecolor,
        facecolor = facecolor,
        linewidth = 1,
        align = 'center',
        zorder = 9,
    )
    ax.errorbar(
        x = x,
        y = [first_extrc,last_extrc],
        yerr = [first_err,last_err],
        color = errcolor,
        linewidth = 2,
        fmt='none',
        zorder = 10,
    )
    return ax
# %%
# SMILEs
fixed_pattern = "decade_mpi"
# %%
# 20CR all ens
CR20_first_extc, CR20_last_extc = read_extrc_rean("CR20_allens")

CR20_first_extc = CR20_first_extc / (4 * 80)
CR20_last_extc = CR20_last_extc / (4 * 80)

# %%
CR20_ens_first_eof, CR20_ens_last_eof = read_eof_rean("CR20")

CR20_ens_first_pc = CR20_ens_first_eof.pc.sel(mode="NAO") - CR20_ens_first_eof.pc.sel(mode = 'NAO').mean()
CR20_ens_last_pc = CR20_ens_last_eof.pc.sel(mode="NAO") - CR20_ens_last_eof.pc.sel(mode = 'NAO').mean()

# also read ensemble mean of 20CR
CR20_ens_first_extc, CR20_ens_last_extc = read_extrc_rean("CR20")
CR20_ens_first_extc = CR20_ens_first_extc / 4
CR20_ens_last_extc = CR20_ens_last_extc / 4


#%%
CR20_composite_ts_first = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/first_composite_mean_ts_40.nc")
CR20_composite_ts_last = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/last_composite_mean_ts_40.nc")
CR20_composite_ts_diff = CR20_composite_ts_last - CR20_composite_ts_first
CR20_composite_ts = xr.concat([CR20_composite_ts_first, CR20_composite_ts_last, CR20_composite_ts_diff], dim="period")
CR20_composite_ts = CR20_composite_ts.ts
CR20_composite_ts['period'] = ["first", "last", "diff"]

#%%
CR20_composite_psl_first = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/first_composite_mean_psl_40.nc")
CR20_composite_psl_last = xr.open_dataset("/work/mh0033/m300883/Tel_MMLE/data/CR20_allens/composite/last_composite_mean_psl_40.nc")
CR20_composite_psl_diff = CR20_composite_psl_last - CR20_composite_psl_first
#%%
# from Pa to hPa
CR20_composite_psl_first = CR20_composite_psl_first.ts / 100
CR20_composite_psl_last = CR20_composite_psl_last.ts / 100
CR20_composite_psl_diff = CR20_composite_psl_diff.ts / 100


#%%
pc_first_df = CR20_ens_first_eof.pc.sel(mode="NAO").to_dataframe().reset_index()
pc_last_df = CR20_ens_last_eof.pc.sel(mode="NAO").to_dataframe().reset_index()

#%%
pc_first_df["period"] = "first"
pc_last_df["period"] = "last"

pc_dfs = pd.concat([pc_first_df[['pc', 'period']], pc_last_df[['pc', 'period']]], ignore_index=True)


#%%
fig = plt.figure(figsize=(180 / 25.4, 180 / 25.4))
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


# first row
ax_hist = fig.add_subplot(331)
ax_pos = fig.add_subplot(332)
ax_neg = fig.add_subplot(333)

# second row
ax_pos_first = fig.add_subplot(334, projection=ccrs.Orthographic(central_longitude=-20, central_latitude=60))
ax_pos_last = fig.add_subplot(335, projection=ccrs.Orthographic(central_longitude=-20, central_latitude=60))
ax_pos_diff = fig.add_subplot(336, projection=ccrs.Orthographic(central_longitude=-20, central_latitude=60))

# third row
ax_neg_first = fig.add_subplot(337, projection=ccrs.Orthographic(central_longitude=-20, central_latitude=60))
ax_neg_last = fig.add_subplot(338, projection=ccrs.Orthographic(central_longitude=-20, central_latitude=60))
ax_neg_diff = fig.add_subplot(339, projection=ccrs.Orthographic(central_longitude=-20, central_latitude=60))


# plot the histograms with 'first' as blue and 'last' as orange, side by side
sns.histplot(
        data=pc_dfs,
        x="pc",
        hue="period",
        hue_order=["first", "last"],
        palette=["#1f77b4", "#ff7f0e"],
        multiple="dodge",
        shrink=0.6,
        bins=np.arange(-4, 4.1, 0.5),
        legend=False,
        ax=ax_hist,
        stat="density",
    )


# line for positive and negative extreme counts


# error bar for 20CR
pos_true_first, pos_err_first, neg_true_first, neg_err_first = y_yerr(CR20_first_extc)
pos_true_last, pos_err_last, neg_true_last, neg_err_last = y_yerr(CR20_last_extc)

pos_true_first_ens, pos_err_first_ens, neg_true_first_ens, neg_err_first_ens = y_yerr(
    CR20_ens_first_extc
)
pos_true_last_ens, pos_err_last_ens, neg_true_last_ens, neg_err_last_ens = y_yerr(
    CR20_ens_last_extc
)

# Bar error bar with "hats" (caps) over the ends
reananlysis_bar(
    pos_true_first,
    pos_err_first,
    pos_true_last,
    pos_err_last,
    ax=ax_pos,
    x=[0.3, 0.8],
    width=0.1,
    facecolor="grey",
    errcolor="gray",
)
# Add caps ("hats") to error bars
ax_pos.errorbar(
    [0.3, 0.8],
    [pos_true_first, pos_true_last],
    yerr=[pos_err_first, pos_err_last],
    fmt='none',
    color='k',
    linewidth=2,
    capsize=6,  # This adds the hats
    zorder=11,
)

reananlysis_bar(
    pos_true_first_ens,
    pos_err_first_ens,
    pos_true_last_ens,
    pos_err_last_ens,
    ax=ax_pos,
    x=[0.2, 0.7],
    width=0.1,
    facecolor="none",
    errcolor="gray",
)
# Add caps ("hats") to error bars
ax_pos.errorbar(
    [0.2, 0.7],
    [pos_true_first_ens, pos_true_last_ens],
    yerr=[pos_err_first_ens, pos_err_last_ens],
    fmt='none',
    color='k',
    linewidth=2,
    capsize=6,
    zorder=11,
)

# negative extremes
reananlysis_bar(
    neg_true_first,
    neg_err_first,
    neg_true_last,
    neg_err_last,
    ax=ax_neg,
    x=[0.3, 0.8],
    width=0.1,
    facecolor="grey",
    errcolor="gray",
)
# Add caps ("hats") to error bars
ax_neg.errorbar(
    [0.3, 0.8],
    [neg_true_first, neg_true_last],
    yerr=[neg_err_first, neg_err_last],
    fmt='none',
    color='k',
    linewidth=2,
    capsize=6,  # This adds the hats
    zorder=11,
)
reananlysis_bar(
    neg_true_first_ens,
    neg_err_first_ens,
    neg_true_last_ens,
    neg_err_last_ens,
    ax=ax_neg,
    x=[0.2, 0.7],
    width=0.1,
    facecolor="none",
    errcolor="gray",
)
# Add caps ("hats") to error bars
ax_neg.errorbar(
    [0.2, 0.7],
    [neg_true_first_ens, neg_true_last_ens],
    yerr=[neg_err_first_ens, neg_err_last_ens],
    fmt='none',
    color='k',
    linewidth=2,
    capsize=6,
    zorder=11,
)

# positive
# ts as color
CR20_composite_ts_first.ts.sel(extr_type = 'pos', mode = 'NAO').plot(
    ax=ax_pos_first,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    levels =np.arange(-1.5, 1.6, 0.3),
    add_colorbar=False,
)

CR20_composite_ts_last.ts.sel(extr_type = 'pos', mode = 'NAO').plot(
    ax=ax_pos_last,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    levels =np.arange(-1.5, 1.6, 0.3),
    extend = 'both',
    add_colorbar=False, 
)
CR20_composite_ts_diff.ts.sel(extr_type = 'pos', mode = 'NAO').plot(
    ax=ax_pos_diff,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    levels =np.arange(-1.5, 1.6, 0.3),
    extend = 'both',
    add_colorbar=False,
)

# psl as lines
CR20_composite_psl_first.sel(extr_type = 'pos', mode = 'NAO').plot.contour(
    ax=ax_pos_first,
    transform=ccrs.PlateCarree(),
    colors="black",
    linewidth=0.5,
    add_colorbar=False,
    levels = [l for l in np.arange(-5, 5.1, 1) if l != 0],
)

CR20_composite_psl_last.sel(extr_type = 'pos', mode = 'NAO').plot.contour(
    ax=ax_pos_last,
    transform=ccrs.PlateCarree(),
    colors="black",
    linewidth=0.5,
    add_colorbar=False,
    levels = [l for l in np.arange(-5, 5.1, 1) if l != 0],
)
CR20_composite_psl_diff.sel(extr_type = 'pos', mode = 'NAO').plot.contour(
    ax=ax_pos_diff,
    transform=ccrs.PlateCarree(),
    colors="black",
    linewidth=0.5,
    add_colorbar=False, 
    levels = [l for l in np.arange(-5, 5.1, 1) if l != 0],
)
# negative

map_temp = CR20_composite_ts_first.ts.sel(extr_type='neg', mode='NAO').plot(
    ax=ax_neg_first,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    levels=np.arange(-1.5, 1.6, 0.3),
    extend='both',
    add_colorbar=False,
)

CR20_composite_ts_last.ts.sel(extr_type='neg', mode='NAO').plot(
    ax=ax_neg_last,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    levels=np.arange(-1.5, 1.6, 0.3),
    extend='both',
    add_colorbar=False,
)

CR20_composite_ts_diff.ts.sel(extr_type='neg', mode='NAO').plot(
    ax=ax_neg_diff,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    levels=np.arange(-1.5, 1.6, 0.3),
    extend='both',
    add_colorbar=False,
)

CR20_composite_psl_first.sel(extr_type='neg', mode='NAO').plot.contour(
    ax=ax_neg_first,
    transform=ccrs.PlateCarree(),
    colors="black",
    linewidth=0.5,
    add_colorbar=False,
    levels=[l for l in np.arange(-5, 5.1, 1) if l != 0],
)

CR20_composite_psl_last.sel(extr_type='neg', mode='NAO').plot.contour(
    ax=ax_neg_last,
    transform=ccrs.PlateCarree(),
    colors="black",
    linewidth=0.5,
    add_colorbar=False,
    levels=[l for l in np.arange(-5, 5.1, 1) if l != 0],
)

CR20_composite_psl_diff.sel(extr_type='neg', mode='NAO').plot.contour(
    ax=ax_neg_diff,
    transform=ccrs.PlateCarree(),
    colors="black",
    linewidth=0.5,
    add_colorbar=False,
    levels=[l for l in np.arange(-5, 5.1, 1) if l != 0],
)




for ax in [ax_pos, ax_neg]:
    ax.set_xlim(-0.2, 1.2)

    # change the x-ticks to first40 and last40
    ax.set_xticks([0.2, 0.8])
    ax.set_xticklabels(["first40", "last40"])

for ax in[ax_hist, ax_pos, ax_neg]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


#### ax2 ####
# add legend
ax_hist.axes.set_facecolor("none")
f_patch_MPI = mpatches.Patch(color="#1f77b4", label="first10")
l_patch_MPI = mpatches.Patch(color="#ff7f0e", label="last10")

ax_hist.set_ylabel(
    "probability density",
)
ax_hist.set_xlabel("NAO index", fontsize=12)

ax_hist.legend(
    handles=[f_patch_MPI, l_patch_MPI],
    loc="lower center",
    frameon=False,
    ncol=2,
    bbox_to_anchor=(0.5, -0.5),
)

empty_patch = mpatches.Patch(
    facecolor="none", edgecolor="black", label="20CR", linewidth=1
)

fill_patch = mpatches.Patch(
    facecolor="grey", edgecolor="black", label="20CR_ens (80)"
)

ax_pos.legend(
    handles=[ empty_patch, fill_patch],
    loc="lower center",
    frameon=False,
    ncol=2,
    bbox_to_anchor=(1.2, -0.5),
)




for ax in [ax_pos_first, ax_pos_last, ax_pos_diff, ax_neg_first, ax_neg_last, ax_neg_diff]:
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    # add gridlines
    ax.gridlines(
        linestyle="--",
    )
    ax.set_title("")


cax = fig.add_axes([0.1, 0.01, 0.8, 0.02])  # [left, bottom, width, height]


fig.colorbar(
    map_temp,
    cax=cax,
    orientation="horizontal",
    aspect=50,
    fraction=0.05,
    pad=0.1,
    shrink=0.8,
    label="near surface temperature anomaly (K)",
)

# add a, b, c, d, e, f, g, h, i labels to the subplots
for i, ax in enumerate(fig.axes[:-1]):
    ax.text(
        -0.1,
        1.0,
        f"{chr(97 + i)}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


plt.tight_layout()
plt.subplots_adjust(
    top=0.95,
    bottom=0.05,
    left=0.05,
    right=0.95,
    hspace=0.35,
    wspace=0.2,
)

# Move the second row (ax_pos_first, ax_pos_last, ax_pos_diff) a bit lower
for ax in [ax_pos_first, ax_pos_last, ax_pos_diff]:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - 0.06, pos.width, pos.height])




# # save the figure
fig.savefig("/work/mh0033/m300883/Tel_MMLE/docs/source/plots/thesis/20CR_allens_nao_extreme.png",
            bbox_inches="tight",
            dpi=300,
            )

#%%


# %%
