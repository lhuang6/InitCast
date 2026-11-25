import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import ctables
from metpy.units import units
import numpy as np
import xarray as xr

from initcast.data.model import Model
from training import Training
import initcast.plot.ct as ct
import initcast.machine as machine

from datetime import datetime, timedelta
import json

def return_product_info(data, product):

    with open("data.json", "r") as f:
        data_json = json.load(f)
        plot_info = data_json['plot']

    norm = None
    cmap = None
    unit = ''

    if product in plot_info:
        unit = plot_info[product]['units']
        if isinstance(product, xr.DataArray):
            data.data = data.data.to(unit)
        else:    
            data = data.to(unit)

        if plot_info[product]['ct'] == 'metpy':
            norm, cmap = ctables.registry.get_with_range(plot_info[product]['ct_name'], plot_info[product]['range'][0], plot_info[product][1])

        if plot_info[product]['ct'] == 'ct':
            if plot_info[product]['ct_name'] == 'dwpt':
                cmap = ct.dwpt_ct()
            if plot_info[product]['ct_name'] == 'composite':
                cmap = ct.composite_ct()
            if plot_info[product]['ct_name'] == 'cin':
                cmap = ct.composite_ct(type='cin')
            if plot_info[product]['ct_name'] == 'wspd':
                cmap = ct.piv_wspd_ct()

            min = plot_info[product]['range'][0]
            max = plot_info[product]['range'][1]
            contours = plot_info[product]['range'][2]

            norm = Normalize(vmin=min, vmax=max) 
            norm.boundaries = np.linspace(min, max, contours)

    return data, norm, cmap, unit

# data = (dataset, longitude, latitude)
# barb_uv = (u, v)
'''
PLOTTING A PRODUCT ON A MAP
-------------------------
Plots a product with matplotlib. Windspeed automatically plots
wind barbs and heights.



Attributes:
    product - name of product to plot:
        - 'refc' - composite reflectivity
        - '2m temp' - 2 meter temperature
        - 'sbcape' - surface-based cape
        - 'mlcape' - mixed-layer cape (180-0 mb above surface)
        - 'sbcin' - surface-based cin
        - '1srh' - 0-1000 meter SRH
        - '3srh' - 0-3000 meter SRH
        - wspd_(300 or 500 or 700 or 850) (ex: 'wspd_300') 
        - wind speed at pressure level
        - hght_(300 or 500 or 700 or 850) (ex: 'hght_300') 
        - height at pressure level

    extent - area in the US to view:
             - [min longitude, max longitude, min latitude, max latitude]
             - 'southeast' - southeast sector

    label - label of plot
    bool_contour = false displays a gridded colormesh, true displays contours

    kwargs:
        uh = overlay uh: "2-5" is 2-5km and "0-3" is 0-3km
'''
def plot_model_data(data, model, norm, cmap, bool_contour=False, label=None, extent=None, **kwargs):
    (dataset, longitude, latitude) = data

    figsize = kwargs.get('figsize', (12, 8))
    product_units = kwargs.get('product_units', '')

    date: str = kwargs.get('date', '')
    forecast = kwargs.get('forecast', '')

    heights = kwargs.get('heights', None)
    heights_levels: int = kwargs.get('heights_levels', None)
    uh = kwargs.get('uh', None)
    barb_uv = kwargs.get('barb_uv', None)
    barb_skip: int = kwargs.get('barb_skip', 30)
    barb_length: int = kwargs.get('barb_length', 6)

    barb_skip = round(((dataset.shape[0] + dataset.shape[1]) / 2) / 15)

    fig = plt.figure(figsize=figsize)
    ax1 = plt.axes(projection=ccrs.PlateCarree())

    if bool_contour:
        product_plot = ax1.contourf(
            longitude,
            latitude,
            dataset,
            levels=norm.boundaries,
            transform=ccrs.PlateCarree(),
            norm=norm,
            cmap=cmap,
        )

    else:
        product_plot = ax1.pcolormesh(
            longitude,
            latitude,
            dataset,
            transform=ccrs.PlateCarree(),
            norm=norm,
            cmap=cmap,
            shading="auto"
        )

    # plots uh contour overlay
    if uh is not None:
        if model == "hrrr":
            ax1.contour(
                uh[1],
                uh[2],
                uh[0],
                transform=ccrs.PlateCarree(),
                levels=[75],
                colors='#4D1F00',
                linewidths=1.5
            )

            ax1.contourf(
                uh[1],
                uh[2],
                uh[0],
                transform=ccrs.PlateCarree(),
                levels=[75, 1e9],
                colors='#4D1F00',
                alpha=0.6
            )

        else:
            print(f"UH contours are not available for the {model} model!")

    # plots heights based on the pressure level
    if heights is not None:
        hght_levels = np.linspace(0, 12000, 301)

        if heights_levels == 300:
            hght_levels = np.linspace(8000, 11000, 101)

        if heights_levels == 500:
            hght_levels = np.linspace(4000, 7000, 101)

        if heights_levels == 700:
            hght_levels = np.linspace(2000, 3500, 51)

        if heights_levels == 850:
            hght_levels = np.linspace(1000, 2200, 41)

        hght_contour = ax1.contour(
            heights[1],
            heights[2],
            heights[0],
            transform=ccrs.PlateCarree(),
            levels=hght_levels,
            colors='black',
            linewidths=1.25,
        )

        if extent is None:
            extent = [latitude.min().item(), latitude.max().item(), longitude.min().item() - 360, longitude.max().item() - 360]

        iy = np.linspace(extent[0], extent[1], 4)
        ix = np.linspace(extent[2], extent[3], 4)

        manual_label = []
        for xi in ix:
            for yi in iy:
                manual_label.append((xi, yi))

        print(manual_label)

        ax1.clabel(hght_contour, fontsize=10, manual=manual_label, inline_spacing=0)

    if barb_uv is not None:

        ax1.barbs(
            longitude[::barb_skip, ::barb_skip],
            latitude[::barb_skip, ::barb_skip],
            barb_uv[0][::barb_skip, ::barb_skip],
            barb_uv[1][::barb_skip, ::barb_skip],
            transform=ccrs.PlateCarree(),
            length=barb_length
        )

    plt.colorbar(product_plot, ax=ax1, label=product_units)
    plt.title(f'{model} {label} ({date}) F{forecast}', fontweight="bold")

    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.STATES, linewidth=0.5)

    if extent is not None:
        ax1.set_extent([extent[2], extent[3], extent[0], extent[1]])

    plt.show()

"""
NEEDS WORK
"""
def plot_model(model: Model, product, bool_contour=False, extent=None, **kwargs):
    if product not in model.fields:
        raise ValueError(f'{product} has not been loaded onto the Model object')

    barbs: int = kwargs.get('barbs', None)
    barb_skip: int = kwargs.get('barb_skip', None)
    barb_length: int = kwargs.get('barb_length', 6)

    uh: str = kwargs.get('uh', None)
    heights: int = kwargs.get('heights', None)

    label = kwargs.get('label', product)

    data = model.retrieve_field(product)[product]
    # field, field_name, _ = model.retrieve_fields(product)
    data, norm, cmap, product_units = return_product_info(data, product)

    uv = None
    if barbs is not None:
        u_v= model.retrieve_field([f'u_{barbs}', f'v_{barbs}'])
        u = u_v['u']
        v = u_v['v']

        uv = (u, v)

        print(v)

    uh_data = None
    if uh is not None:
        uh= model.retrieve_field([f'{uh} uh'])
        uh_data = (uh, uh.longitude, uh.latitude)

    heights_data = None
    if heights is not None:
        heights_field = model.retrieve_field([f'hght_{heights}'])
        heights_data = (heights_field, heights_field.longitude, heights_field.latitude)

    plot_model_data((data, data.longitude, data.latitude), model.model, norm, cmap, bool_contour=bool_contour,
                    label=label, extent=extent, date=model.date, forecast=model.forecast, product_units=product_units, barb_uv=uv,
                    barb_skip=barb_skip, barb_length=barb_length, uh=uh_data, heights=heights_data, heights_levels=heights)

# barbs is level of barbs to plot
def plot_input(training: Training, hour, product, bool_contour=False, extent=None, **kwargs):
    with open("data.json", "r") as f:
        data = json.load(f)
        herbie = data["herbie"]
        calc_units = data["calculate_units"]

    barbs: int = kwargs.get('barbs', None)
    barb_skip: int = kwargs.get('barb_skip', None)
    barb_length: int = kwargs.get('barb_length', 6)

    label = kwargs.get('label', product)

    product_index = training.return_feature_index(product)

    X, _ = training.extract_hour(hour)
    X_product = X[:, product_index].reshape(training.grid_shape)

    if product in herbie:
        X_product *= units(herbie[product]['units'])
    else:
        X_product *= units(calc_units[product])

    uv = None
    if barbs is not None:
        u_index = training.return_feature_index(f'u_{barbs}')
        v_index = training.return_feature_index(f'v_{barbs}')

        u = (X[:, u_index].reshape(training.grid_shape) * units('m/s')).to('knots')
        v = (X[:, v_index].reshape(training.grid_shape) * units('m/s')).to('knots')

        uv = (u, v)

    data, norm, cmap, product_units = return_product_info(X_product, product)

    plot_model_data((data, training.longitude, training.latitude), 'rap', norm, cmap, bool_contour=bool_contour,
                    label=label, extent=extent, date=hour, forecast=0, product_units=product_units, barb_uv=uv,
                    barb_skip=barb_skip, barb_length=barb_length)

# hour: YYYY-MM-DD HH:MM
def plot_output(output, bool_contour=False, extent=None, **kwargs):

    (y, longitude, latitude) = output

    cmap = kwargs.get('cmap', 'viridis')
    contour_color = kwargs.get('contour_color', 'grey')
    figsize: tuple = kwargs.get('figsize', (12, 8))
    label = kwargs.get('label', 'Initiation')
    hour = kwargs.get('hour', '')

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    if extent is not None:
        ax.set_extent([extent[2], extent[3], extent[0], extent[1]])

    if bool_contour:
        plot = ax.contourf(longitude,
                           latitude,
                           y,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap,
                           vmin=0,
                           vmax=1,
                           levels=np.linspace(0, 1, 11))

        ax.contour(longitude,
                   latitude,
                   y,
                   transform=ccrs.PlateCarree(),
                   colors=contour_color,
                   vmin=0,
                   vmax=1,
                   levels=np.linspace(0, 1, 11),
                   alpha=0.5,
                   linewidth=0.6)

    else:
        plot = ax.pcolormesh(longitude,
                           latitude,
                           y,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap,
                           vmin=0,
                           vmax=1)

    plt.colorbar(plot, ax=ax)
    plt.title(f'{label} ({hour})', fontweight="bold")

def plot_target(training: Training, start_hour, hours: int, **kwargs):

    target = training.composite_target(start_hour, hours)
    target = target.reshape(training.grid_shape)

    plot_output((target, training.longitude, training.latitude), label=f'{hours}-Hour Target Initiation',
                     hour=start_hour, **kwargs)

def plot_predict_prob(test: Training, model, start_hour, hours: int, **kwargs):
    probs, _ = machine.composite_pred(test, model, start_hour, hours)

    probs = probs.reshape(test.grid_shape)

    plot_output((probs, test.longitude, test.latitude), label=f'{hours}-Hour Probability of Initiation',
                     hour=start_hour, **kwargs)