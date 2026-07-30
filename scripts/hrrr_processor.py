#!/usr/bin/env python3
"""
HRRR Data Processor for Project Vorticity
Runs at 01Z/07Z/13Z/19Z to process 00Z/06Z/12Z/18Z HRRR runs respectively.
Generates plots and uploads to GitHub for web viewer consumption.

Usage:
    python hrrr_processor.py [--run YYYYMMDD_HHz] [--output-dir DIR] [--github-upload]

Environment Variables:
    GITHUB_TOKEN - GitHub personal access token for pushing to repo
    GITHUB_REPO - Repository in format "owner/repo" (default: projectvorticity/hrrr-plots)
"""

import os
import sys
import datetime
import argparse
import warnings
import subprocess
from pathlib import Path

import numpy as np
import requests
import cfgrib
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Products to generate
PRODUCTS = [
    "sbcape", "via_meso", "dewpoint", "stp", "srh_01km",
    "reflectivity", "hazard", "cig_contours",
    "wind_500mb", "wind_700mb", "wind_850mb",
    "high_risk_disc"
]

SECTORS = {
    "full": (-125, -65, 22, 52),
    "northeast": (-85, -65, 38, 48),
    "southeast": (-95, -75, 28, 38),
    "plains": (-105, -85, 32, 46),
    "midwest": (-95, -80, 36, 46),
    "southwest": (-120, -105, 30, 40),
    "northwest": (-125, -110, 40, 50),
}

HRRR_PROJ = ccrs.LambertConformal(
    central_longitude=-97.5,
    central_latitude=30,
    standard_parallels=(30, 60)
)

# ==================== COLOR MAPS ====================
CAPE_BOUNDS = [0, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000]
CAPE_COLORS = ["#f0f0f0", "#d0f0d0", "#a0f0a0", "#50c850", "#00a000", "#ffff00",
               "#ffc000", "#ff8000", "#ff4000", "#ff0000", "#c00000"]

DEWPOINT_BOUNDS = [-30, -20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
DEWPOINT_COLORS = ["#E5E5FF", "#B2B2FF", "#8080FF", "#4D4DFF", "#1919FF", "#00CCCC",
                   "#00A6A6", "#008080", "#00FF00", "#FFFF00", "#FFD700", "#FFA500",
                   "#FF7F00", "#FF2A00", "#E50000"]

VIA_BOUNDS = [0, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25]
VIA_COLORS = [
    "#FFFFFF00", "#A4DAF5", "#BFE5A8", "#FFFF54", "#FDF260", "#F5C242",
    "#F1A041", "#EA3323", "#EA334C", "#D86DCD", "#DAA1D9", "#ECCFED",
    "#A175F7", "#9E59F6", "#7D2AF5", "#600FE2", "#0000A5", "#000000", "#000000"
]

STP_BOUNDS = [0, 0.5, 1, 2, 3, 4, 5, 7, 10, 11, 15]
STP_COLORS = ["#ffffff00", "#a0f0a0", "#00a000", "#ffff00", "#ffc000", "#ff8000",
              "#ff4000", "#ff0000", "#c00000", "#800080"]

SRH_BOUNDS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 2000]
SRH_COLORS = ["#ffffff00", "#99CCFF", "#66B2FF", "#3399FF", "#0080FF", "#0066CC",
              "#0052A3", "#003D7A", "#002952", "#001429", "#000A14", "#000A14", "#000A14"]

REFC_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
REFC_COLORS = ["#ffffff00", "#00ecec", "#01a0f6", "#0000f6", "#00ff00", "#00c800",
               "#009000", "#ffff00", "#e7c000", "#ff9000", "#ff0000", "#d60000",
               "#c00000", "#ff00ff", "#9955c9", "#000000"]

HAZ_BOUNDS = [0, 1, 2, 3, 4, 5, 6]
HAZ_COLORS = ["#ffffff00", "#add8e6", "#ffff00", "#ffcccb", "#8b0000", "#ff69b4"]

# Wind speed colormaps for upper-level winds
WIND_BOUNDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
WIND_COLORS = ["#ffffff00", "#d0f0d0", "#a0f0a0", "#50c850", "#00a000", "#ffff00",
               "#ffc000", "#ff8000", "#ff4000", "#ff0000", "#c00000", "#800080"]

# ==================== LOGGING ====================
def log(msg: str):
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[{timestamp}] {msg}")


# ==================== HRRR DATA DOWNLOAD ====================
def download_hrrr_file(date_str: str, run_hour: int, fxx: int, file_type: str, out_path: str) -> bool:
    """Download a single HRRR file from AWS."""
    base_url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus"
    if file_type == "sfc":
        url = f"{base_url}/hrrr.t{run_hour:02d}z.wrfsfcf{fxx:02d}.grib2"
    else:
        url = f"{base_url}/hrrr.t{run_hour:02d}z.wrfnatf{fxx:02d}.grib2"
    try:
        log(f"Downloading {os.path.basename(out_path)}...")
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code != 200:
            log(f"Failed to download (HTTP {response.status_code})")
            return False
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        log(f"Downloaded {os.path.basename(out_path)}")
        return True
    except Exception as e:
        log(f"Error downloading: {e}")
        return False


def get_hrrr_grid(ds):
    """Extract lat/lon grid from HRRR dataset."""
    if 'longitude' in ds.coords:
        lon = ds.longitude.values
        if np.any(lon > 180):
            lon = np.where(lon > 180, lon - 360, lon)
        if lon.ndim == 1 and 'latitude' in ds.coords:
            lat = ds.latitude.values
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lon2d, lat2d
        elif lon.ndim == 2:
            return lon, ds.latitude.values
    for var in ds.data_vars:
        if hasattr(ds[var], 'longitude') and hasattr(ds[var], 'latitude'):
            lon = ds[var].longitude.values
            lat = ds[var].latitude.values
            if lon.ndim == 1:
                lon2d, lat2d = np.meshgrid(lon, lat)
                return lon2d, lat2d
            else:
                return lon, lat
    raise ValueError("Could not extract lat/lon from HRRR dataset")


# ==================== DATA EXTRACTION ====================
def get_sbcape(all_ds, shape2d):
    """Extract Surface-Based CAPE from datasets."""
    best_data = None
    best_max = -1
    for ds in all_ds:
        for var_name in list(ds.data_vars.keys()):
            if any(c in var_name.lower() for c in ['cape', 'cape_surface']):
                try:
                    data = ds[var_name].values.astype(float)
                    if data.ndim == 3:
                        data = data[0]
                    if data.ndim != 2:
                        data = data.reshape(shape2d)
                    data_max = np.nanmax(data)
                    if data_max > best_max:
                        best_data = data
                        best_max = data_max
                except Exception:
                    pass
    return best_data if best_data is not None else np.zeros(shape2d)


def get_variable(data_dict, keys, default_shape):
    """Get a variable from data dictionary with fallback."""
    for k in keys:
        if k in data_dict:
            v = data_dict[k].values.astype(float)
            if v.ndim == 2:
                return v
            elif v.ndim == 3:
                v = v[0]
                if v.shape == default_shape:
                    return v
    return np.zeros(default_shape)


def get_pressure_level_wind(data_dict, level_hPa, shape2d):
    """Extract wind at a specific pressure level."""
    u_data = np.zeros(shape2d)
    v_data = np.zeros(shape2d)
    if 'u_3d' in data_dict and 'v_3d' in data_dict:
        u_3d = data_dict['u_3d']
        v_3d = data_dict['v_3d']
        try:
            if 'isobaricInhPa' in u_3d.dims:
                u_data = u_3d.sel(isobaricInhPa=level_hPa, method='nearest').values
                v_data = v_3d.sel(isobaricInhPa=level_hPa, method='nearest').values
            else:
                levels = u_3d.coords.get('isobaricInhPa', u_3d.coords.get('level', None))
                if levels is not None:
                    idx = np.argmin(np.abs(levels.values - level_hPa))
                    u_data = u_3d.isel(isobaricInhPa=idx).values if 'isobaricInhPa' in u_3d.dims else u_3d.isel(
                        level=idx).values
                    v_data = v_3d.isel(isobaricInhPa=idx).values if 'isobaricInhPa' in v_3d.dims else v_3d.isel(
                        level=idx).values
            if u_data.ndim == 3:
                u_data = u_data[0]
            if v_data.ndim == 3:
                v_data = v_data[0]
        except Exception as e:
            log(f"Error extracting {level_hPa}mb wind: {e}")
    return u_data, v_data


# ==================== PLOTTING ====================
def plot_filled_contour(lons, lats, data, cmap, norm, title, out_file, meta_text="",
                        sector="full", cig1=None, cig2=None, cig3=None, uh_data=None):
    """Plot filled contour map."""
    fig = plt.figure(figsize=(16, 12), dpi=100)
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.90], projection=HRRR_PROJ)
    extent = SECTORS.get(sector, SECTORS["full"])
    ax.set_extent(extent)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    if data is not None:
        im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.6)
        cbar.ax.tick_params(labelsize=10)
    if cig1 is not None and np.any(cig1):
        ax.contour(lons, lats, cig1, levels=[0.5], colors='black', linewidths=2.0, transform=ccrs.PlateCarree())
    if cig2 is not None and np.any(cig2):
        ax.contour(lons, lats, cig2, levels=[0.5], colors='black', linewidths=2.5, transform=ccrs.PlateCarree())
        ax.contourf(lons, lats, cig2, levels=[0.5, 1.5], colors=['gray'], alpha=0.25, transform=ccrs.PlateCarree())
    if cig3 is not None and np.any(cig3):
        ax.contour(lons, lats, cig3, levels=[0.5], colors='black', linewidths=3.0, transform=ccrs.PlateCarree())
        ax.contourf(lons, lats, cig3, levels=[0.5, 1.5], colors=['black'], alpha=0.5, transform=ccrs.PlateCarree())
    if uh_data is not None and np.any(uh_data >= 50):
        ax.contour(lons, lats, uh_data, levels=[50, 100, 200],
                   colors=['purple', 'red', 'black'], linewidths=2.5, transform=ccrs.PlateCarree())
    fig.text(0.5, 0.97, title, ha='center', va='center', fontsize=24, weight='bold')
    if meta_text:
        fig.text(0.03, 0.03, meta_text, ha='left', va='bottom', fontsize=12, color='black', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))
    plt.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.close('all')
    log(f"Saved {out_file}")


def plot_wind_with_barbs(lons, lats, u_data, v_data, title, out_file, meta_text="",
                         sector="full", barb_skip=20):
    """Plot wind speed with wind barbs overlay."""
    fig = plt.figure(figsize=(16, 12), dpi=100)
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.90], projection=HRRR_PROJ)
    extent = SECTORS.get(sector, SECTORS["full"])
    ax.set_extent(extent)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)
    wind_speed_ms = np.sqrt(u_data ** 2 + v_data ** 2)
    wind_speed_kts = wind_speed_ms * 1.94384
    cmap = mcolors.ListedColormap(WIND_COLORS)
    norm = mcolors.BoundaryNorm(WIND_BOUNDS, len(WIND_COLORS))
    im = ax.pcolormesh(lons, lats, wind_speed_kts, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=40, shrink=0.6)
    cbar.set_label('Wind Speed (knots)', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    u_kts = u_data * 1.94384
    v_kts = v_data * 1.94384
    ax.barbs(lons[::barb_skip, ::barb_skip], lats[::barb_skip, ::barb_skip],
             u_kts[::barb_skip, ::barb_skip], v_kts[::barb_skip, ::barb_skip],
             transform=ccrs.PlateCarree(), length=5, barbcolor='black',
             flagcolor='black', linewidth=0.5)
    fig.text(0.5, 0.97, title, ha='center', va='center', fontsize=24, weight='bold')
    if meta_text:
        fig.text(0.03, 0.03, meta_text, ha='left', va='bottom', fontsize=12, color='black', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))
    plt.savefig(out_file, dpi=100, bbox_inches='tight')
    plt.close('all')
    log(f"Saved {out_file}")


# ==================== MAIN PROCESSING ====================
def process_forecast_hour(date_str: str, run_hour: int, fxx: int, output_dir: Path, sector: str = "full"):
    """Process a single forecast hour and generate all products."""
    log(f"Processing {date_str} {run_hour:02d}Z F{fxx:02d} for sector {sector}")
    run_dir = output_dir / f"runs/{date_str}_{run_hour:02d}z"
    run_dir.mkdir(parents=True, exist_ok=True)
    sfc_file = output_dir / f"temp_sfc_{date_str}_{run_hour:02d}z_F{fxx:02d}.grib2"
    nat_file = output_dir / f"temp_nat_{date_str}_{run_hour:02d}z_F{fxx:02d}.grib2"
    sfc_ok = download_hrrr_file(date_str, run_hour, fxx, "sfc", str(sfc_file))
    nat_ok = download_hrrr_file(date_str, run_hour, fxx, "nat", str(nat_file))
    if not sfc_ok:
        log(f"Failed to download surface file for F{fxx:02d}")
        return False
    try:
        ds_sfc = cfgrib.open_datasets(str(sfc_file), backend_kwargs={'errors': 'ignore'})
        ds_nat = cfgrib.open_datasets(str(nat_file), backend_kwargs={'errors': 'ignore'}) if nat_ok else []
        all_ds = ds_sfc + ds_nat
        data_dict = {}
        for ds in all_ds:
            for var in ds.data_vars:
                if var in ['u', 'v', 'gh'] and 'isobaricInhPa' in ds.dims:
                    data_dict[f"{var}_3d"] = ds[var]
                elif var not in data_dict:
                    data_dict[var] = ds[var]
        lons, lats = None, None
        for ds in all_ds:
            try:
                lons, lats = get_hrrr_grid(ds)
                break
            except:
                continue
        if lons is None:
            raise ValueError("No valid lat/lon found")
        shape2d = lons.shape
        SBCAPE = get_sbcape(all_ds, shape2d)
        PRES = get_variable(data_dict, ['sp', 'PRES'], shape2d)
        TMP = get_variable(data_dict, ['t2m', 'TMP', '2t'], shape2d)
        DPT = get_variable(data_dict, ['d2m', 'DPT', '2d'], shape2d)
        U10 = get_variable(data_dict, ['u10', 'UGRD'], shape2d)
        V10 = get_variable(data_dict, ['v10', 'VGRD'], shape2d)
        SBCINH = get_variable(data_dict, ['cin', 'CIN'], shape2d)
        REFC = get_variable(data_dict, ['refc', 'REFC'], shape2d)
        UPHL = get_variable(data_dict, ['uphe', 'UPHL'], shape2d)
        HLCY = get_variable(data_dict, ['hlcy', 'HLCY'], shape2d)
        u_500, v_500 = get_pressure_level_wind(data_dict, 500, shape2d)
        u_700, v_700 = get_pressure_level_wind(data_dict, 700, shape2d)
        u_850, v_850 = get_pressure_level_wind(data_dict, 850, shape2d)

        TMP700 = np.zeros(shape2d)

        for ds in all_ds:
            if 't' not in ds.data_vars:
                continue

            if 'isobaricInhPa' not in ds['t'].dims:
                continue

            try:
                TMP700 = (
                        ds['t']
                        .sel(isobaricInhPa=700, method='nearest')
                        .squeeze()
                        .values
                        - 273.15
                )
                break

            except Exception:
                continue
        # LCL
        try:
            p_flat = PRES.flatten() * units.Pa
            T_flat = TMP.flatten() * units.kelvin
            Td_flat = DPT.flatten() * units.kelvin
            lcl_p, _ = mpcalc.lcl(p_flat, T_flat, Td_flat)
            h_lcl_flat = mpcalc.pressure_to_height_std(lcl_p) - mpcalc.pressure_to_height_std(p_flat)
            LCL = np.clip(h_lcl_flat.to('m').magnitude.reshape(PRES.shape), 0, None)
            LCL = np.nan_to_num(LCL, nan=0.0)
        except:
            LCL = np.zeros_like(PRES)
        # Storm motion
        if 'ustm' in data_dict and 'vstm' in data_dict:
            USTM = data_dict['ustm'].values
            VSTM = data_dict['vstm'].values
            if USTM.ndim > 2: USTM = USTM[0]
            if VSTM.ndim > 2: VSTM = VSTM[0]
        else:
            USTM = U10 + 5.0
            VSTM = V10
        # Derived parameters
        SRW_01 = np.sqrt((U10 - USTM) ** 2 + (V10 - VSTM) ** 2)
        Sw_01 = HLCY / (np.where(SRW_01 == 0, 1e-5, SRW_01) * 1000.0)
        via_mask = (SRW_01 > 5) & (Sw_01 > 0.0025) & (LCL > 0) & (LCL < 2500)
        CAPE_TERM = np.where(SBCAPE < 2000, SBCAPE / 1500.0, (SBCAPE ** 2) / 3000000.0)
        via_val = np.sqrt(np.clip((SRW_01 - 5) / 5.0, 0, None) * ((100.0 * Sw_01 - 0.25) ** 2) * CAPE_TERM * (
                    (2500.0 - LCL) / 1000.0))
        VIA_MESO = np.where(via_mask, via_val, 0)
        DPT_C = DPT - 273.15
        BWD_06_ms = np.sqrt((u_500 - U10) ** 2 + (v_500 - V10) ** 2)
        cape_term = SBCAPE / 1500.0
        lcl_term = np.maximum(0.0, (2000.0 - LCL) / 1000.0)
        srh_term = HLCY / 150.0
        bwd_term = BWD_06_ms / 20.0
        STP = cape_term * lcl_term * srh_term * bwd_term
        HIGH_RISK_DISC = np.where(STP > 1, STP - TMP700, -0.001)
        SRW_kts = SRW_01 * 1.94384
        BWD_06_kts = BWD_06_ms * 1.94384
        HAZARD = np.zeros_like(SBCAPE)
        pds_mask = (STP >= 3) & (HLCY >= 200) & (SRW_kts >= 15) & (BWD_06_kts >= 45) & (LCL < 1000) & (SBCINH < 50)
        tor_mask = ((STP >= 1) & (SBCINH < 125)) & ~pds_mask
        mrgl_tor_mask = ((STP >= 0.5) & (HLCY >= 150) & (SBCINH < 150)) & ~tor_mask & ~pds_mask
        svr_mask = ((STP >= 1) | (SBCAPE >= 1000)) & (SBCINH < 50) & ~mrgl_tor_mask & ~tor_mask & ~pds_mask
        mrgl_svr_mask = (SBCAPE >= 500) & (SBCINH < 75) & ~svr_mask & ~mrgl_tor_mask & ~tor_mask & ~pds_mask
        HAZARD[mrgl_svr_mask] = 1
        HAZARD[svr_mask] = 2
        HAZARD[mrgl_tor_mask] = 3
        HAZARD[tor_mask] = 4
        HAZARD[pds_mask] = 5
        local_max_stp = maximum_filter(STP, size=15)
        local_max_via = maximum_filter(VIA_MESO, size=15)
        local_max_refc = maximum_filter(REFC, size=15)
        cig1 = (local_max_stp > 2) & (local_max_via > 2) & ((HAZARD == 3) | (HAZARD == 4) | (HAZARD == 5))
        cig2 = (local_max_stp > 4) & (local_max_via > 5) & ((HAZARD == 4) | (HAZARD == 5))
        cig3 = (local_max_stp > 9) & (local_max_via > 7) & (local_max_refc > 50) & (HAZARD == 5)
        # Colormaps
        c_cmap = mcolors.ListedColormap(CAPE_COLORS)
        c_norm = mcolors.BoundaryNorm(CAPE_BOUNDS, len(CAPE_COLORS))
        v_cmap = mcolors.ListedColormap(VIA_COLORS)
        v_norm = mcolors.BoundaryNorm(VIA_BOUNDS, len(VIA_COLORS))
        d_cmap = mcolors.ListedColormap(DEWPOINT_COLORS)
        d_norm = mcolors.BoundaryNorm(DEWPOINT_BOUNDS, len(DEWPOINT_COLORS))
        s_cmap = mcolors.ListedColormap(STP_COLORS)
        s_norm = mcolors.BoundaryNorm(STP_BOUNDS, len(STP_COLORS))
        h_cmap = mcolors.ListedColormap(SRH_COLORS)
        h_norm = mcolors.BoundaryNorm(SRH_BOUNDS, len(SRH_COLORS))
        r_cmap = mcolors.ListedColormap(REFC_COLORS)
        r_norm = mcolors.BoundaryNorm(REFC_BOUNDS, len(REFC_COLORS))
        hz_cmap = mcolors.ListedColormap(HAZ_COLORS)
        hz_norm = mcolors.BoundaryNorm(HAZ_BOUNDS, len(HAZ_COLORS))
        valid_dt = datetime.datetime.strptime(date_str, "%Y%m%d") + datetime.timedelta(hours=run_hour + fxx)
        meta_text = f"HRRR {date_str} {run_hour:02d}Z\nF{fxx:02d} Valid: {valid_dt.strftime('%Y-%m-%d %H')}Z\nSector: {sector.upper()}"
        for sec_name in SECTORS.keys():
            sec_suffix = sec_name
            plot_filled_contour(lons, lats, SBCAPE, c_cmap, c_norm, "SBCAPE (J/kg)",
                                run_dir / f"sbcape_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_filled_contour(lons, lats, VIA_MESO, v_cmap, v_norm, "VIA-MESO Index",
                                run_dir / f"via_meso_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_filled_contour(lons, lats, DPT_C, d_cmap, d_norm, "2m Dewpoint (C)",
                                run_dir / f"dewpoint_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_filled_contour(lons, lats, STP, s_cmap, s_norm, "Significant Tornado Parameter",
                                run_dir / f"stp_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_filled_contour(lons, lats, HLCY, h_cmap, h_norm, "0-1km Storm Relative Helicity",
                                run_dir / f"srh_01km_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_filled_contour(lons, lats, REFC, r_cmap, r_norm, "Composite Reflectivity",
                                run_dir / f"reflectivity_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name,
                                uh_data=UPHL)
            plot_filled_contour(lons, lats, HAZARD, hz_cmap, hz_norm, "Psbl. Hazard Type",
                                run_dir / f"hazard_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_filled_contour(lons, lats, None, None, None, "CIG Contours",
                                run_dir / f"cig_contours_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name,
                                cig1=cig1, cig2=cig2, cig3=cig3)
            plot_high_risk_discriminator(
                lons,
                lats,
                HIGH_RISK_DISC,
                "High Risk Discriminator (STP - 700mb Temp)",
                run_dir / f"high_risk_disc_{sec_suffix}_F{fxx:02d}.png",
                meta_text,
                sec_name
            )
            plot_wind_with_barbs(lons, lats, u_500, v_500, "500mb Wind & Barbs",
                                 run_dir / f"wind_500mb_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_wind_with_barbs(lons, lats, u_700, v_700, "700mb Wind & Barbs",
                                 run_dir / f"wind_700mb_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
            plot_wind_with_barbs(lons, lats, u_850, v_850, "850mb Wind & Barbs",
                                 run_dir / f"wind_850mb_{sec_suffix}_F{fxx:02d}.png", meta_text, sec_name)
        log(f"Successfully processed F{fxx:02d}")
        return True
    except Exception as e:
        log(f"Error processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for f in [sfc_file, nat_file]:
            if f.exists():
                f.unlink()

def plot_high_risk_discriminator(
        lons,
        lats,
        data,
        title,
        out_file,
        meta_text="",
        sector="full"):

    fig = plt.figure(figsize=(16,12), dpi=100)

    ax = fig.add_axes(
        [0.02,0.05,0.96,0.90],
        projection=HRRR_PROJ
    )

    ax.set_extent(SECTORS.get(sector, SECTORS["full"]))

    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.STATES.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    neg_levels = [-3,-2,-1]

    pos_levels = [1,2,3]

    ax.contour(
        lons,
        lats,
        data,
        levels=neg_levels,
        colors='dodgerblue',
        linewidths=1,
        transform=ccrs.PlateCarree()
    )

    ax.contour(
        lons,
        lats,
        data,
        levels=pos_levels,
        colors='gold',
        linewidths=1,
        transform=ccrs.PlateCarree()
    )

    ax.contour(
        lons,
        lats,
        data,
        levels=[0],
        colors='red',
        linewidths=1.2,
        transform=ccrs.PlateCarree()
    )

    fig.text(
        0.5,
        0.97,
        title,
        ha='center',
        fontsize=24,
        weight='bold'
    )

    if meta_text:
        fig.text(
            0.03,
            0.03,
            meta_text,
            fontsize=12,
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.9
            )
        )

    plt.savefig(out_file, dpi=100, bbox_inches='tight')

    plt.close('all')

    log(f"Saved {out_file}")

def upload_to_github(output_dir: Path, date_str: str, run_hour: int):
    """Upload generated images to GitHub (only if there are new files)."""
    if not GITHUB_TOKEN:
        log("GITHUB_TOKEN not set, skipping upload")
        return False
    run_dir = output_dir / f"runs/{date_str}_{run_hour:02d}z"
    if not run_dir.exists():
        log(f"Run directory not found: {run_dir}")
        return False
    try:
        # Initialize git repo if needed
        if not (output_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=output_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin",
                            f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"],
                           cwd=output_dir, check=True)
            subprocess.run(["git", "config", "user.email", "bot@projectvorticity.com"], cwd=output_dir, check=True)
            subprocess.run(["git", "config", "user.name", "HRRR Bot"], cwd=output_dir, check=True)
        # Check for changes
        result = subprocess.run(["git", "status", "--porcelain"], cwd=output_dir, capture_output=True, text=True)
        if not result.stdout.strip():
            log("No new files to upload, skipping")
            return True
        # Add and commit
        subprocess.run(["git", "add", f"runs/{date_str}_{run_hour:02d}z"], cwd=output_dir, check=True)
        subprocess.run(["git", "commit", "-m", f"Update HRRR {date_str} {run_hour:02d}Z plots (partial)"],
                       cwd=output_dir, check=True)
        subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=output_dir, check=True)
        log("Successfully uploaded to GitHub")
        return True
    except subprocess.CalledProcessError as e:
        log(f"Git operation failed: {e}")
        return False


def cleanup_old_runs(output_dir: Path, hours_to_keep: int = 60):
    """Remove runs older than specified hours."""
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours_to_keep)
    for run_path in runs_dir.iterdir():
        if run_path.is_dir():
            try:
                parts = run_path.name.split('_')
                date_str = parts[0]
                hour_str = parts[1].replace('z', '')
                run_dt = datetime.datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
                if run_dt < cutoff:
                    log(f"Removing old run: {run_path.name}")
                    import shutil
                    shutil.rmtree(run_path)
            except Exception:
                pass


def get_latest_available_run():
    """Find the most recent available HRRR run among 00,06,12,18Z."""
    now = datetime.datetime.utcnow()
    synoptic_hours = [0, 6, 12, 18]
    for i in range(48):
        dt = now - datetime.timedelta(hours=i)
        hour = dt.hour
        if hour not in synoptic_hours:
            continue
        date_str = dt.strftime("%Y%m%d")
        run_hour = hour
        url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/hrrr.t{run_hour:02d}z.wrfsfcf00.grib2.idx"
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return date_str, run_hour
        except:
            pass
    return None, None


def main():
    parser = argparse.ArgumentParser(description="HRRR Data Processor for Project Vorticity")
    parser.add_argument("--run", type=str, help="Specific run in format YYYYMMDD_HHz (e.g., 20240315_12z)")
    parser.add_argument("--output-dir", type=str, default="./hrrr_output", help="Output directory")
    parser.add_argument("--github-upload", action="store_true", help="Upload to GitHub every 3 forecast hours")
    parser.add_argument("--forecast-hours", type=str, default="0-48", help="Forecast hours range (e.g., 0-48)")
    parser.add_argument("--cleanup", action="store_true", help="Remove runs older than 60 hours")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run:
        parts = args.run.split('_')
        date_str = parts[0]
        run_hour = int(parts[1].replace('z', ''))
    else:
        date_str, run_hour = get_latest_available_run()
        if date_str is None:
            log("No available HRRR runs found")
            sys.exit(1)

    log(f"Processing HRRR run: {date_str} {run_hour:02d}Z")

    fhr_parts = args.forecast_hours.split('-')
    start_fhr = int(fhr_parts[0])
    end_fhr = int(fhr_parts[1]) if len(fhr_parts) > 1 else start_fhr

    success_count = 0
    processed_count = 0
    for fxx in range(start_fhr, end_fhr + 1):
        if process_forecast_hour(date_str, run_hour, fxx, output_dir):
            success_count += 1
            processed_count += 1
            log(f"Uploading after {processed_count} forecast hours")
            upload_to_github(output_dir, date_str, run_hour)

    log(f"Processed {success_count}/{end_fhr - start_fhr + 1} forecast hours")

    if args.github_upload and processed_count > 0:
        log("Final upload after all forecast hours")
        upload_to_github(output_dir, date_str, run_hour)

    if args.cleanup:
        cleanup_old_runs(output_dir, hours_to_keep=60)

    log("Processing complete")


if __name__ == "__main__":
    main()
