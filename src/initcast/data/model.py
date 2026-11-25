"""
Class for loading all the data and information of a forecast hour of the HRRR or RAP model.
"""

import initcast.plot.ct as ct

from herbie import Herbie
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import ctables
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
from matplotlib.colors import Normalize

import json
from importlib import resources
import warnings

class Model:
    """
    Stores all the data and information of a forecast hour of the HRRR or RAP model.

    Args:
        model: ('rap' or 'hrrr') which weather model specified
        date: (YYYY-MM-DD HH:MM) date and hour of the model run 
        forecast: forecast hour

    Attributes:
        model: ('rap' or 'hrrr') which weather model specified
        date: (YYYY-MM-DD HH:MM) date and hour of the model run 
        forecast: forecast hour
        self.dx: horizontal resolution of the model
        loaded_sounding: if the sounding method has been called before
        pressure_name: name of the data for downloading pressure levels with Herbie
        pressure_lvls: Herbie object that stores all the pressure levels
        field_list: dictionary of all fields retrievable by Herbie for the model
        calculate_fields: dictionary 
        fields: dictionary storing all model fields as an xarray grid

    Raises:
        ValueError: if model is not 'hrrr' or 'rap'
    """

    def __init__(self, model: str, date: str, forecast: int):
        self.date = date
        self.forecast = forecast
        self.loaded_sounding = False
        self.model = model
        self.dx = 13000 * units.m

        # different name for pressure levels based on the model
        if model == "hrrr":
            self.pressure_name = "prs"
            self.dx = 3000

        elif model == "rap":
            self.pressure_name = "awp130pgrb"

        else:
            raise ValueError(f"Only 'hrrr' and 'rap' are implemented. Got {model}")

        # creates and stores Herbie object
        self.pressure_lvls = Herbie(
            self.date,
            model=self.model,
            product=self.pressure_name,
            fxx=self.forecast
        )

        # empty dictionary to store all fields of the model
        self.fields = {}

        # opening up the data from the data.json file
        data_json = (resources.files("initcast") / "data").joinpath("data.json")

        with data_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
            self.field_list = data["herbie"]          # dictionary of all the fields from the model
            self.calculate_fields = data["calculate"] # all the fields needed to be calculated

    def display_loaded_fields(self):
        """
        Prints all the fields already loaded into the object.
        """

        # Printing out each key in the fields dictionary
        for key in self.fields:
            print(key)

    def get_herbie(self, list):
        """
        To-do"""

        temp_input_fields = {}

        # adding all the fields that can be searched with Herbie
        for field in list:
            # using 'search' to find the field with Herbie
            # 'search' is based on the model's idx file that comes with the grib
            xarray = self.pressure_lvls.xarray(rf"{self.field_list[field]["search"]}")
            temp_input_fields[field] = xarray[self.field_list[field]['xarray_name']] * units(self.field_list[field]['units'])

        self.fields.update(temp_input_fields)

        return temp_input_fields

    def calculate(self, name):
        """
        To-do"""
        warnings.filterwarnings("ignore")

        # checking if it is something that can be calculated
        if name not in self.calculate_fields:
            return ValueError(f"{name} is not a valid feature to calculate! The valid features are: {self.calculate_features}")

        # default u and v names and units
        u_name = "u"
        v_name = "v"

        # checking if the feature has a pressure level
        if "_" in name:
            if ("wspd" in name) or ("vort" in name) or ("va" in name) or ("advection" in name):
                # level is the millibars based off the number after "_"
                level = name.split("_")[-1]

                # all fieldnames in the data.json file are (field)_(pressure) unless at the surface
                u_name = f"u_{level}"
                v_name = f"v_{level}"
                vort_name = f"vort_{level}"

        # checking if the u and v components have already been retrieved before
        if u_name not in self.fields:
            self.get_herbie([u_name])
        if v_name not in self.fields:
            self.get_herbie([v_name])

        u = self.fields[u_name]
        v = self.fields[v_name]

        # pythagorean theorem for total windspeed
        if "wspd" in name or name == "shear_6000":
            parameter = (u ** 2 + v ** 2) ** 0.5

        # divergence (or convergence)
        if name == "sfc_conv":
            parameter = mpcalc.divergence(u, v, dx=self.dx, dy=self.dx)

        # vorticity
        if "vort_" in name:
            parameter = mpcalc.vorticity(u, v, dx=self.dx, dy=self.dx)

        # vorticity advection
        if "va_" in name:

            # checking if vorticity has already been loaded in 
            if vort_name in self.fields:
                vort = self.fields[vort_name]
            else:
                # retrieving vorticity if not already
                vort = self.get_herbie([vort_name])[vort_name]
            
            parameter = mpcalc.advection(vort, u, v, dx=self.dx, dy=self.dx)

        # differential positive vorticity advection (500-300mb)
        if "dpva" in name:

            # va_list is both 500mb and 300mb vorticity advection xarray arrays
            va_list = []

            for va_name in "va_500", "va_300":
                # checking if 500mb or 300mb vorticity advection has already been retrieved
                if va_name in self.fields:
                    va = self.fields[va_name]
                # adding in vorticity advection if not already retrieved before
                else:
                    # name of vorticity in json
                    vort_name = va_name.replace("va", "vort")

                    # checking if vorticity has already been retrieved
                    if vort_name in self.fields:
                        vort = self.fields[vort_name]
                    # if not, then calculating vorticity
                    else:
                        # name of u and v 
                        u_name = va_name.replace("va", "u")
                        v_name = va_name.replace("va", "v")

                        # checking if 500mb or 300mb u and v have been retireved
                        if u_name in self.fields and v_name in self.fields:
                            u = self.fields[u_name]
                            v = self.fields[v_name]
                        else:
                            # retrieving u and v if not already in there
                            uv = self.get_herbie([u_name, v_name])
                            u = uv[u_name]
                            v = uv[v_name]

                        # calculating vorticity
                        vort = mpcalc.vorticity(u, v, dx=self.dx, dy=self.dx) 
                    # calculating vorticity advection
                    va = mpcalc.advection(vort, u, v, dx=self.dx, dy=self.dx)
                # adding 500mb or 300mb vorticity advection
                va_list.append(va)

            # finding dpva
            parameter = va_list[0] - va_list[1]

        # checking if temperature has already been retrieved for these fields
        if "advection" in name or "q_vect_conv" in name:

            if "advection" in name:
                temp_name = f"temp_{level}"
            if "q_vect_conv" in name:
                temp_name = "temp_850"

            if temp_name not in self.fields:
                t = self.get_herbie([temp_name])[temp_name]
            else:
                t = self.fields[temp_name]

        # temperature advection
        if "advection_" in name:
            parameter = mpcalc.advection(t, u, v, dx=self.dx, dy=self.dx)

        # Q vector convergence
        if name == "q_vect_conv":
            q_vect = mpcalc.q_vector(u, v, t, 850 * units['hPa'], dx=self.dx, dy=self.dx)
            parameter = mpcalc.divergence(q_vect[0], q_vect[1], dx=self.dx, dy=self.dx)

        ds = parameter.to_dataset(name=name)
        ds = ds[name]

        # adding the feature to the fields dictionary in the class
        feature = { 
            name: ds
        }

        self.fields.update(feature)

        return feature

    def retrieve_field(self, list):
        """To-do"""

        # sorts between fields that can be accessed directly and calculated
        # dictionary of all the fields that the model needs to retrieve
        temp_input_fields = {}

        # list of all the fields that the model needs to calculate
        temp_calculate_fields = []

        # sorting all the fields
        for arg in list:
            if arg in self.field_list:
                print(f'getting {arg}..')
                temp_input_fields.update(self.get_herbie([arg]))
            elif arg in self.calculate_fields:
                temp_calculate_fields.append(arg)
            else:
                print(f"{arg} is not a valid feature!")

        # ordering the list so fields wont be calculated multiple times
        ordered_calc = sorted(temp_calculate_fields, key=lambda x: self.calculate_fields.index(x))

        # calculating all the fields
        for calc in ordered_calc:
            calc_feature = self.calculate(calc)

            temp_input_fields.update(calc_feature)

        return temp_input_fields    

    '''
    RETRIEVING MODEL FIELDS
    -------------------------
    Retrieves and returns fields from the model.
    
    
    
    Attributes:
        *args - fields to retrieve:
            - 'refc' - composite reflectivity
            - '2m temp' - 2 meter temperature
            - '2m dewp' - 2 meter dewpoint temperature
            - 'sbcape' - surface-based cape
            - 'mlcape' - mixed-layer cape (180-0 mb above surface)'
            - 'sbcin' - surface-based cin
            - 'mlcin' - mixed-layer cin (180-0 mb above surface)'
            - '2-5 uh' - 2-5km updraft helicity
            - '0-3 uh' - 0-3km updraft helicity
            - '1srh' - 0-1000 meter SRH
            - '3srh' - 0-3000 meter SRH
            - 
            - wspd_(pressure in millibars) (ex: 'wspd_300') 
                - wind speed at pressure level
                - 1000, 975, 950, 925, 900, 875... 100 mb are all available (25mb intervals)
            - hght_(pressure in millibars) (ex: 'hght_300') 
                - height at pressure level
                - 1000, 975, 950, 925, 900, 875... 100 mb are all available (25mb intervals)
        
        add_uv (bool) - whether or not to include u and v components of wind in the output when 
                        calculating wind speed
        
        
        
    Returns (field_list, field_names, field_units):
        Returns a list of xarray datasets, xarray variable names,
        and the units for the datasets. If uv is True, the order
        of the list will be u then v then wspd.
        
        field_list - xarray datasets of the field in the order 
                       of *args
        field_names - list of name of the xarray dataset variable 
                        (ex. ds[field_names])
        field_units - units of the dataset which aren't explicitly
                        stated in the dataset
    '''
    def retrieve_fields(self, *args, add_uv=False):
        field_list = []
        field_names = []
        field_units = []

        # appends the return lists with an xarray dataset based on the *args
        for field in args:

            if field == "refc":
                field_list.append(self.pressure_lvls.xarray(r"REFC:entire atmosphere:"))
                field_names.append('refc')
                field_units.append('dBz')
            if field == "2m dewp":
                field_list.append(self.pressure_lvls.xarray(r"DPT:2 m above ground:"))
                field_names.append('d2m')
                field_units.append('K')
            if field == "2m temp":
                field_list.append(self.pressure_lvls.xarray(r"TMP:2 m above ground:"))
                field_names.append('t2m')
                field_units.append('K')
            if field == "sbcape":
                field_list.append(self.pressure_lvls.xarray(r"CAPE:surface:"))
                field_names.append('cape')
                field_units.append('J/kg')
            if field == "mlcape":
                field_list.append(self.pressure_lvls.xarray(r"CAPE:180-0 mb above ground:"))
                field_names.append('cape')
                field_units.append('J/kg')
            if field == "sbcin":
                field_list.append(self.pressure_lvls.xarray(r"CIN:surface:"))
                field_names.append('cin')
                field_units.append('J/kg')
            if field == "mlcin":
                field_list.append(self.pressure_lvls.xarray(r"CIN:180-0 mb above ground:"))
                field_names.append('cin')
                field_units.append('J/kg')

            if field == '2-5 uh':
                field_list.append(self.pressure_lvls.xarray(r"MXUPHL:5000-2000 m above ground"))
                field_names.append('unknown')
                field_units.append('J/kg')

            if field == '0-3 uh':
                field_list.append(self.pressure_lvls.xarray(r"MXUPHL:3000-0 m above ground"))
                field_names.append('unknown')
                field_units.append('J/kg')

            # all SRH fields, but only '1srh' and '3srh' are available
            if 'srh' in field:
                srh_height = field.replace('srh', '')

                field_list.append(self.pressure_lvls.xarray(rf"HLCY:{srh_height}000-0 m above ground:"))
                field_names.append('hlcy')
                field_units.append('J/kg')

            # all wspd fields, but only 300, 500, 700, and 850 available
            if ('wspd_' in field) or ('shear_' in field):
                u = None
                v = None
                u_name = None
                v_name = None

                if 'shear_6000' in field:
                    uv = self.pressure_lvls.xarray(r'VVCSH:0-6000 m above ground')

                    u_name = 'vucsh'
                    v_name = 'vvcsh'

                    u = uv['vucsh']
                    v = uv['vvcsh']

                elif "wspd_" in field:
                    wspd_pressure = field.replace('wspd_', '')

                    if wspd_pressure == 'sfc':
                        wspd_level = '10 m above ground'
                        u_name = 'u10'
                        v_name = 'v10'
                    else:
                        wspd_level = wspd_pressure + " mb"
                        u_name = 'u'
                        v_name = 'v'

                    if self.model == 'hrrr':
                        u = self.pressure_lvls.xarray(rf"UGRD:{wspd_level}")['u']
                        v = self.pressure_lvls.xarray(rf"VGRD:{wspd_level}")['v']
                    if self.model == 'rap':
                        uv = self.pressure_lvls.xarray(rf"VGRD:{wspd_level}")
                        u = uv[u_name]
                        v = uv[v_name]

                if (u is None) or (v is None):
                    raise ValueError(f'{field} is not a valid field')

                # appends u and v if uv is True
                if add_uv:
                    field_list.append(u.to_dataset())
                    field_list.append(v.to_dataset())

                    field_names.append(u_name)
                    field_names.append(v_name)

                    field_units.append('m/s')
                    field_units.append('m/s')

                field_list.append((np.hypot(u, v)).to_dataset(name="wspd"))
                field_names.append('wspd')
                field_units.append('m/s')

            if 'hght_' in field:
                hght_pressure = field.replace('hght_', '')
                field_list.append(self.pressure_lvls.xarray(rf"HGT:{hght_pressure} mb:"))
                field_names.append('gh')
                field_units.append('m')

            # if only one argument was passed, then it returns just the object
            if len(field_list) == 1:
                field_list = field_list[0]
                field_names = field_names[0]
                field_units = field_units[0]

        return field_list, field_names, field_units

    '''
    RETURN COORDINATES
    -------------------------
    Returns the (x, y) coordinate of the closest gridpoint to any latitude and longitude

    Attributes:
        lat - latitude of the sounding (ex. 21.5)
        lon - longitude of the sounding (ex. -87.2)
        dataset - optional, grid used to find the coordinates
    '''
    def return_coords(self, lat, lon, dataset=None):
        if dataset is None:
            dataset = self.pressure_lvls.xarray(r"HGT:500 mb:")

        lon_360 = lon + 360
        dist = np.sqrt((dataset["latitude"] - lat) ** 2 + (dataset["longitude"] - lon_360) ** 2)
        y, x = np.unravel_index(np.argmin(dist.values), dist.shape)

        return x, y

    '''
    PLOT A SOUNDING
    -------------------------
    Plots a sounding with the sounderpy library. It may take a little bit to load
    the pressure data when first using the method in a script, but once loaded,
    the pressure fields are stored in memory. 
    
    Attributes:
        lat - latitude of the sounding (ex. 21.5)
        longitude - longitude of the sounding (ex. -87.2)
    '''
    def sounding(self, lat, lon):

        try:
            # import sounderpy to plot sounding
            import sounderpy as spy
        except ImportError as err:
            raise ImportError(
                "sounding requires sounderpy. "
                "Install it with: pip install initcast[soundings]"
            ) from err

        # if a sounding has not been loaded yet
        if not self.loaded_sounding:

            # searching up each product
            self.hgt = self.pressure_lvls.xarray(r"HGT:\d+ mb")
            self.u = self.pressure_lvls.xarray(r"UGRD:\d+ mb")
            self.v = self.pressure_lvls.xarray(r"VGRD:\d+ mb")
            self.temp = self.pressure_lvls.xarray(r"TMP:\d+ mb")
            self.dewp = self.pressure_lvls.xarray(r"DPT:\d+ mb")

            self.loaded_sounding = True

        # find closest gridpoint to the location
        lon_360 = lon + 360
        dist = np.sqrt((self.hgt["latitude"] - lat) ** 2 + (self.hgt["longitude"] - lon_360) ** 2)
        y, x = np.unravel_index(np.argmin(dist.values), dist.shape)

        clean_data = {}

        # turning each product into a vertical profile and converting the units
        vtemp = ((self.temp["t"].isel(x=x, y=y).to_numpy()) * units.kelvin).to('degC')
        vdewp = ((self.dewp['dpt'].isel(x=x, y=y).to_numpy()) * units.kelvin).to('degC')
        vu = (self.u['u'].isel(x=x, y=y).to_numpy() * (units.m / units.s)).to('knots')
        vv = (self.v['v'].isel(x=x, y=y).to_numpy() * (units.m / units.s)).to('knots')
        vprs = self.u["isobaricInhPa"].to_numpy() * units.hPa
        vhgt = (self.hgt['gh'].isel(x=x,y=y).to_numpy()) * units.m

        # puts all vertical profiles into a dictionary for sounderpy to read
        clean_data['p'] = vprs
        clean_data['z'] = vhgt
        clean_data['T'] = vtemp
        clean_data['Td'] = vdewp
        clean_data['u'] = vu
        clean_data['v'] = vv

        # turns self.date into a list containing the seperate year, month, day, etc.
        split_day = self.date.split(" ")[0].split('-')
        split_time = self.date.split(" ")[1]

        # creates clean_data dictionary
        clean_data['site_info'] = {
            'site-id': 'none',
            'site-name': 'none',
            'site-lctn': 'none',
            'site-latlon': [lat, lon_360],
            'site-elv': 0,
            'source': 'none',
            'model': 'HRRR',
            'fcst-hour': self.forecast,
            'run-time': ['none', 'none', 'none', 'none'],
            'valid-time': [split_day[0], split_day[1], split_day[2], split_time]}

        # titles for the sounding
        clean_data['titles'] = {
            'top_title': 'HRRR FORECAST',
            'left_title': 'INIT: ' + self.date + 'HOUR ' + str(self.forecast),
            'right_title': f'[{lat}, {lon}]    '}

        # builds sounding using sounderpy function
        spy.build_sounding(clean_data, save=True)