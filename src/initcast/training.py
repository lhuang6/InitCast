"""
TRAINING CLASS
-------------------------
This object stores all the features and output required to train or create a Machine-Learning
model that predicts convective initiation. Input and output for each hour is saved as a numpy
file under the directory /training/. /X/ is for input, and y is for output.



Class variables:
    self.model = model used for the training data
    self.extent = extent [min lat, max lat, min lon, max lon] of the data in the CONUS
    self.min_lat, self.max_lat, self.min_lon, self.max_lon = extent of the data as
                                                             individual variables
    self.reference_grid = 'Model' Object used to find information of the grid
    self.longitude = numpy array of the longitude values in every gridpoint
    self.latitude = numpy array of the latitude values in every gridpoint
    self.grid_shape = shape (tuple) of the entire grid
    self.grid_size = total number of gridpoints
    self.x = numpy array of all the x values
    self.y = numpy array of all the y values
    self.feature_names = list of all the names of the features used for the object
    self.days = list of all the hours used
    self.data = data used to train a model or run a model with
    self.target = target output to train a model with




Methods:
    array_to_dataset(self, array, scale=1):
        converts a numpy array of the training data into an xarray Dataset

    date_range(start, end, input_format="%Y-%m-%d", output_format="%Y-%m-%d"):
        returns a list of range of all days specified

    create_output(self, day, hour, return_all=False, xarray=True, **kwargs):
        finds where convective initiation has occurred at any day and hour

    add_training(self, days: list, **kwargs):
        add a new list of days to the existing training data in the object

    extract_hour(self, hour):
        returns the input and output of 1 hour from the training data

    composite_target(self, start_hour, hours: int):
        returns a composite of all the outputs of several hours

    return_feature_index(self, feature):
        finds the index of a feature
"""

from initcast.data.model import Model
import initcast.mrms as mrms

import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import pysteps
import scipy

import warnings
from datetime import datetime, timedelta
import os
import glob
import json

class Training:

    def array_to_dataset(self, array, scale=1):
        ds = xr.Dataset(
            data_vars={
                "label": (("y", "x"), array)
            },

            coords={
                "latitude": (("y", "x"), self.latitude[::scale, ::scale]),
                "longitude": (("y", "x"), self.longitude[::scale, ::scale]),
            }
        )

        return ds

    def mrms_to_dataset(self, radar, reference_grid=None, scale=1):
        if reference_grid is None:
            latitude = radar.latitude
            longitude = radar.longitude
        else:
            latitude = reference_grid.latitude
            longitude = reference_grid.longitude

        ds = xr.Dataset(
            data_vars={
                "label": (("latitude", "longitude"), radar)
            },

            coords={
                "latitude": latitude[::scale],
                "longitude": longitude[::scale]
            }
        )

        return ds

    @staticmethod
    def date_range(start, end, input_format="%Y-%m-%d", output_format="%Y-%m-%d"):
        start_date = datetime.strptime(start, input_format)
        end_date = datetime.strptime(end, input_format)

        return [
            (start_date + timedelta(days=i)).strftime(output_format)
            for i in range((end_date - start_date).days + 1)
        ]

    """
    ESTIMATING NEW INITIATION
    -------------------------
    Returns a 0.04 degree boolean numpy array (0 or 1) that calculates new convective 
    initiation based on MRMS radar data.


    The method to labeling new convective initiation is as followed:

    First, an area of >40 dbz is marked to estimate where a "storm" might be. To
    identify whether it is new, every few minutes, the future radar is predicted
    using optical flow storm motion estimation. The optical flow method used in
    this method is the Lucas-Kanade (LK) algorithm using the PySTEPS library. This 
    is then compared with the actual future radar to find if a storm is newly 
    initiated or has just moved. A small margin of error is used 
    (boolean dilation) in case the storm has changed directions or has expanded. 

    Lastly, it is checked with an echotops threshold of 10km to make sure
    the intiation identified is convective.




    Attributes:
        day (string) - YYYYMMDD in UTC
        hour (string) - HH in UTC
        extent - tuple [min_lon, max_lon, min_lat, max_lat]

        return_all - if true, return all composite reflectivity, estimated 
                     (extrapolated) composite reflectivity, echotops boolean lists
        repeat - length of each interval in minutes to calculate storm motion
        inbetween - how many files inbetween to add to storm motion algorithm
        iter - how many iterations to perform binary dilation



    Returns (initiation):
        initiation - boolean numpy array where 1 is a gridpoint where a new storm has
                     initiated at any time in the hour

        return_all = True (initiation, initiation_list, echotops_list, R_last_list, R_extrap_list):
            initiation_list - list of initiation numpy arrays for every timestep
            echotops_list - list of echotop boolean arrays for every timestep
            R_last_list - list of composite reflectivity boolean arrays for every timestep
            R_extrap_list - list of the estimated composite reflectivity boolean arrays for every timestep 
    """

    def create_output(self, day, hour, return_all=False, xarray=True, **kwargs):
        repeat: int = kwargs.get('repeat', 20)
        inbetween: int = kwargs.get('inbetween', 1)
        iter: int = kwargs.get('iter', 3)
        between_hour: bool = kwargs.get('between_hour', True)

        if repeat > 60:
            raise ValueError('Repeat minutes must be less than or equal to an hour')

        if 60 % repeat != 0:
            raise ValueError('Repeat minutes must be divisible by the hour')

        if repeat % 2 != 0:
            raise ValueError('Repeat minutes must be even')

        if repeat % inbetween != 0:
            raise ValueError('Repeat minutes must be divisible by the amount of files inbetween')

        if repeat / inbetween % 2 != 0:
            raise ValueError('Repeat minutes must be divisible by the amount of files inbetween')

        # how many mrms files to skip to download
        et_skip = int(repeat / 2)
        cref_skip = int(et_skip / (inbetween + 1))

        # how many timesteps inputted into each LK motion estimation
        timesteps = int(et_skip / cref_skip)

        print(f"et_skip: {et_skip}, cref_skip: {cref_skip}")

        # longitude converted into 360 degrees
        min_lon = 360 + self.min_lon
        max_lon = 360 + self.max_lon

        # importer reads each mrms file
        # oflow_method is the LK storm motion calculation
        # extrap_method estimates future radar based on the LK storm motion
        importer = pysteps.io.get_method('mrms_grib', 'importer')
        oflow_method = pysteps.motion.get_method("LK")
        extrap_method = pysteps.extrapolation.interface.get_method('semilagrangian')

        et_files = mrms.download_data(day, hour, 'echotops', return_files=True, skip=int(et_skip),
                                      between_hour=between_hour)
        cref_files = mrms.download_data(day, hour, 'cref', return_files=True, skip=int(cref_skip),
                                        between_hour=between_hour)

        # lists that outputs the initiation, echotops, reflectivity, and extrapolated reflectivity for each time step
        numpy_initiation_list = []
        initiation_list = []
        echotops_list = []
        R_last_list = []
        R_extrap_list = []

        # loop to read every file at every time step
        # minute tracks the minutes in the loop
        frames = []
        minute = 0
        for i, file in enumerate(cref_files):

            # reads each file and appends it to a list
            arr, _, metadata = importer(file, extent=[min_lon, max_lon, self.min_lat, self.max_lat], window_size=4)
            frames.append(arr)

            # computes LK storm motion algorithm
            # computes every time it has passed the amount of repeat minutes
            if minute != 0 and minute % repeat == 0:
                et_file_index = int(minute / repeat)

                # uses xarray to read echotops file
                echotops_orig = xr.open_dataset(et_files[et_file_index], engine='cfgrib')
                echotops_orig = echotops_orig.sel(
                    latitude=slice(self.max_lat, self.min_lat),
                    longitude=slice(min_lon, max_lon))

                # scales down to 4km
                echotops_4km = pysteps.utils.dimension.aggregate_fields(echotops_orig.unknown.values, window_size=(4, 4),
                                                                        axis=(0, 1))

                # stacks the list of composite reflectivity scans for oflow_method() to read
                R = np.stack(frames)
                R_last = R[-1, :, :].copy()
                R_first = R[0, :, :].copy()

                R, metadata = pysteps.utils.transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)
                V1 = oflow_method(R)

                # estimates where the storms from the first scan will end up in the last scan using the LK storm motion
                R_extrap = extrap_method(R_first, V1, timesteps, allow_nonfinite_values=True)

                # creates booleans for the echotops and composite reflectivity
                R_last = xr.where(R_last > 40, 1, 0)
                R_last_extrap = xr.where(R_extrap[timesteps - 1] > 40, 1, 0)
                echotops_4km = xr.where(echotops_4km > 10, 1, 0)

                # creates a small margin of error to account for a storm to expand
                R_last_extrap = scipy.ndimage.binary_dilation(R_last_extrap, iterations=iter)

                # boolean where high reflectivity not associated with a moving storm has high enough echotops
                # estimates places where storms have initiated
                init = np.minimum(echotops_4km, R_last & ~R_last_extrap)

                if xarray:
                    numpy_initiation_list.append(init)
                    echotops_list.append(self.mrms_to_dataset(echotops_4km, reference_grid=echotops_orig, scale=4))
                    R_last_list.append(self.mrms_to_dataset(R_last, reference_grid=echotops_orig, scale=4))
                    R_extrap_list.append(self.mrms_to_dataset(R_last_extrap, reference_grid=echotops_orig, scale=4))
 
                else:
                    initiation_list.append(init)
                    echotops_list.append(echotops_4km)
                    R_last_list.append(R_last)
                    R_extrap_list.append(R_last_extrap)

                # resets list of reflectivity frames
                # adds last reflectivity scan to act as the first reflectivity scan for the next time the LK model is computed
                frames = []
                frames.append(arr)

                print(f'{minute} minute timestep complete')

            minute += cref_skip * 2

        if xarray:
            initiation = np.maximum.reduce(numpy_initiation_list)
            initiation = self.mrms_to_dataset(initiation, reference_grid=echotops_orig, scale=4)
        else:
            initiation = np.maximum.reduce(initiation_list)

        if return_all:
            return initiation, initiation_list, echotops_list, R_last_list, R_extrap_list
        else:
            return initiation
        
    """
    SCALING OUTPUT (INITIATION) TO MODEL GRID
    -----------------------------------------
    This method scales output or initiation data onto the training model grid. If any gridpoint in the new grid contains 
    a point of initiation, then the whole new gridpoint is marked as a point of initiation.

    Attributes:
        label - Boolean numpy array from create_output() marking places of initiation

    Returns:
        scaled_label - Boolean numpy array in the resolution of the model used for the object
    """
    def scale_output(self, label):
        # new blank label array with same size as model grid
        scaled_label = np.zeros(self.grid_shape, dtype=np.uint8)

        # ---------DOWNSCALING MRMS INITIATION GRID TO MODEL GRID---------
        # checking every cell in the model grid for initiation
        # any model cell with one initiation point is marked as 1
        for y in self.y:
            for x in self.x:

                # latitude and longitude of the center of the cell
                center_lat = self.latitude[y, x].item()
                center_lon = self.longitude[y, x].item()

                # finds the distance from the center to the edge of the cell
                # looks at a surrounding cell's position and divides by 2 to find edge
                if y == (self.reference_grid.sizes['y'] - 1):
                    n_edge = (center_lat - self.latitude[y - 1, x].item()) / 2
                else:
                    n_edge = (self.latitude[y + 1, x].item() - center_lat) / 2

                if y == 0:
                    s_edge = (self.latitude[y + 1, x].item() - center_lat) / 2
                else:
                    s_edge = (center_lat - self.latitude[y - 1, x].item()) / 2

                if x == (self.reference_grid.sizes['x'] - 1):
                    e_edge = (center_lon - self.longitude[y, x - 1].item()) / 2
                else:
                    e_edge = (self.longitude[y, x + 1].item() - center_lon) / 2

                if x == 0:
                    w_edge = (self.longitude[y, x + 1].item() - center_lon) / 2
                else:
                    w_edge = (center_lon - self.longitude[y, x - 1].item()) / 2

                # boundaries of the model cell
                n_bndry = center_lat + n_edge
                s_bndry = center_lat - s_edge
                w_bndry = center_lon - w_edge
                e_bndry = center_lon + e_edge

                # finds all the MRMS label points inside the model cell
                # gives a margin of 0.02 for the edges of the MRMS cells
                latitudes = []
                latitudes_index = []
                for i, lat in enumerate(label.latitude.values):
                    if (n_bndry + 0.02) > lat > (s_bndry - 0.02):
                        latitudes.append(lat)
                        latitudes_index.append(i)

                longitudes = []
                longitudes_index = []
                for i, lon in enumerate(label.longitude.values):
                    if (e_bndry + 0.02) > lon > (w_bndry - 0.02):
                        longitudes.append(lon)
                        longitudes_index.append(i)

                # checks all the MRMS cells to see if at least one of them is 1
                # if one of the MRMS cells are 1, then the model cell is 1
                stop = False
                for mrms_y in latitudes_index:
                    for mrms_x in longitudes_index:
                        if int(label['label'][mrms_y, mrms_x]) == 1:
                            scaled_label[y, x] = 1
                            stop = True
                            break
                    if stop:
                        break

                warnings.filterwarnings("ignore")

        return scaled_label
    
    """
    CREATING INPUT DATA
    -------------------------
    This method creates all of the input data needed. All features inputted are based off
    of the field "feature_names". 

    Attributes:
        M: Model - Model object to retrieve the fields from

        Other keyword arguments:
            features (default: feature_names) - List of features to input. Must come from data.json

    Returns:
        X - Numpy array of the input data which can train a model directly
    
    """
    def create_input(self, M: Model, **kwargs):
        features = kwargs.get('features', self.feature_names)
        input_features = M.retrieve_field(features)

        feature_list = []
        for feature in features:
            ds = input_features[feature]
            ds = ds.sel(x=slice(self.min_x, self.max_x), 
                   y=slice(self.min_y, self.max_y))
            
            feature_list.append(ds.values.flatten())

        X = np.column_stack(tuple(feature_list))

        return X

        """sbcape = M.retrieve_fields('sbcape')[0]
        mlcape = M.retrieve_fields('mlcape')[0]
        mucape = M.pressure_lvls.xarray(r'CAPE:255-0 mb above ground')
        mucin = M.pressure_lvls.xarray(r'CIN:255-0 mb above ground')
        sbcin = M.retrieve_fields('sbcin')[0]
        mlcin = M.retrieve_fields('mlcin')[0]
        lfc = M.pressure_lvls.xarray(r'HGT:level of free convection')
        pbl = M.pressure_lvls.xarray(r'HGT:planetary boundary layer')
        li = M.pressure_lvls.xarray(r'LFTX:500-1000 mb')

        temp = M.retrieve_fields('t2m')[0]
        dewp = M.retrieve_fields('d2m')[0]
        dew_dep = M.pressure_lvls.xarray(r'DEPR:2 m above ground')

        u, v, wspd = M.retrieve_fields('sfcwspd', add_uv=True)[0]
        u_500, v_500, wspd_500 = M.retrieve_fields('500wspd', add_uv=True)[0]
        u_300, v_300, wspd_300 = M.retrieve_fields('300wspd', add_uv=True)[0]
        u_700, v_700, wspd_700 = M.retrieve_fields('700wspd', add_uv=True)[0]
        u_850, v_850, wspd_850 = M.retrieve_fields('850wspd', add_uv=True)[0]

        # 0-6km shear
        ushear_6000, vshear_6000, shear_6000 = M.retrieve_fields('0-6km shear', add_uv=True)[0]

        # vertical velocity, temperature, and relative humidity
        other_1000 = M.pressure_lvls.xarray(r'(VVEL|TMP|RH):1000 mb:')
        other_850 = M.pressure_lvls.xarray(r'(VVEL|TMP|RH):850 mb:')
        other_700 = M.pressure_lvls.xarray(r'(VVEL|TMP|RH):700 mb:')
        other_500 = M.pressure_lvls.xarray(r'(VVEL|TMP|RH):500 mb:')
        other_300 = M.pressure_lvls.xarray(r'(VVEL|TMP|RH):300 mb:')

        input_list = [sbcape, sbcin, mlcape, mlcin, mucape, mucin,
                      temp, dewp, lfc, pbl, li, dew_dep, u, v, wspd,
                      u_500, v_500, wspd_500, u_300, v_300, wspd_300,
                      u_700, v_700, wspd_700, u_850, v_850, wspd_850,
                      ushear_6000, vshear_6000, shear_6000,
                      other_1000, other_850, other_700, other_500,
                      other_300]

        # cuts the data to the extent in the CONUS provided
        input_list = [
            ds.sel(x=slice(self.min_x, self.max_x), y=slice(self.min_y, self.max_y))
            for ds in input_list
        ]

        (sbcape, sbcin, mlcape, mlcin, mucape, mucin,
         temp, dewp, lfc, pbl, li, dew_dep, u, v, wspd,
         u_500, v_500, wspd_500, u_300, v_300, wspd_300,
         u_700, v_700, wspd_700, u_850, v_850, wspd_850,
         ushear_6000, vshear_6000, shear_6000,
         other_1000, other_850, other_700, other_500,
         other_300) = input_list

        # flattening all the data into a 1D array

        sbcape = sbcape['cape'].values.flatten()
        sbcin = sbcin['cin'].values.flatten()
        mlcape = mlcape['cape'].values.flatten()
        mlcin = mlcin['cin'].values.flatten()
        mucape = mucape['cape'].values.flatten()
        mucin = mucin['cin'].values.flatten()
        temp = temp['t2m'].values.flatten()
        dewp = dewp['d2m'].values.flatten()
        dew_dep = dew_dep['unknown'].values.flatten()
        lfc = lfc['gh'].values.flatten()
        pbl = pbl['gh'].values.flatten()
        li = li['lftx'].values.flatten()

        # attempt at finding boundaries
        divergence = mpcalc.divergence(u['u10'] * units('m/s'), v['v10'] * units('m/s'),
                                       dx=13000 * units('m'), dy=13000 * units('m')).values.flatten()

        advection_850 = mpcalc.advection(other_850['t'], u_850['u'], v_850['v'],
                                         dx=13000 * units('m'), dy=13000 * units('m')).values.flatten()
        advection_700 = mpcalc.advection(other_700['t'], u_700['u'], v_700['v'],
                                         dx=13000 * units('m'), dy=13000 * units('m')).values.flatten()
        advection_500 = mpcalc.advection(other_500['t'], u_500['u'], v_500['v'],
                                         dx=13000 * units('m'), dy=13000 * units('m')).values.flatten()

        vort_700 = mpcalc.vorticity(u_700['u'], v_700['v'], dx=13000 * units('m'), 
                                    dy=13000 * units('m')).values.flatten()
        va_700 = mpcalc.advection(vort_700, u_700['u'], v_700['v'], dx=(13000 * units.m), dy=(13000 * units.m)) * 1e5

        vort_500 = mpcalc.vorticity(u_500['u'], v_500['v'], dx=13000 * units('m'), 
                                            dy=13000 * units('m')).values.flatten()
        va_500 = mpcalc.advection(vort_500, u_500['u'], v_500['v'], dx=(13000 * units.m), dy=(13000 * units.m)) * 1e5

        vort_300 = mpcalc.vorticity(u_300['u'], v_300['v'], dx=13000 * units('m'), 
                                                    dy=13000 * units('m')).values.flatten()
        va_300 = mpcalc.advection(vort_300, u_300['u'], v_300['v'], dx=(13000 * units.m), dy=(13000 * units.m)).values.flatten() * 1e5
        dpva = va_300 - va_500

        q_vect = mpcalc.q_vector(u_850['u'], v_850['v'], other_850['t'], 850 * units.hPa, dx=(13000 * units.m), dy=(13000 * units.m))
        q_vect_conv = mpcalc.diverence(q_vect[0], q_vect[1], dx=(13000 * units.m), dy=(13000 * units.m)).values.flatten()


        wspd = wspd['wspd'].values.flatten()
        u = u['u10'].values.flatten()
        v = v['v10'].values.flatten()
        wspd_500 = wspd_500['wspd'].values.flatten()
        u_500 = u_500['u'].values.flatten()
        v_500 = v_500['v'].values.flatten()
        wspd_300 = wspd_300['wspd'].values.flatten()
        u_300 = u_300['u'].values.flatten()
        v_300 = v_300['v'].values.flatten()
        wspd_700 = wspd_700['wspd'].values.flatten()
        u_700 = u_700['u'].values.flatten()
        v_700 = v_700['v'].values.flatten()
        wspd_850 = wspd_850['wspd'].values.flatten()
        u_850 = u_850['u'].values.flatten()
        v_850 = v_850['v'].values.flatten()
        ushear_6000 = ushear_6000['vucsh'].values.flatten()
        vshear_6000 = vshear_6000['vvcsh'].values.flatten()
        shear_6000 = shear_6000['wspd'].values.flatten()

        vvel_1000 = other_1000['w'].values.flatten()
        vvel_850 = other_850['w'].values.flatten()
        vvel_700 = other_700['w'].values.flatten()
        vvel_500 = other_500['w'].values.flatten()
        vvel_300 = other_300['w'].values.flatten()

        temp_1000 = other_1000['t'].values.flatten()
        temp_850 = other_850['t'].values.flatten()
        temp_700 = other_700['t'].values.flatten()
        temp_500 = other_500['t'].values.flatten()
        temp_300 = other_300['t'].values.flatten()

        rh_1000 = other_1000['r'].values.flatten()
        rh_850 = other_850['r'].values.flatten()
        rh_700 = other_700['r'].values.flatten()
        rh_500 = other_500['r'].values.flatten()
        rh_300 = other_300['r'].values.flatten()

        # turns all the variables into a bunch of columns
        input_hour = np.column_stack((sbcape, sbcin, mlcape, mlcin, mucape, mucin,
                                      temp, dewp, lfc, pbl, li, dew_dep, divergence, u,
                                      v, wspd, u_500, v_500, wspd_500, u_300, v_300,
                                      wspd_300, u_700, v_700, wspd_700, u_850, v_850,
                                      wspd_850, ushear_6000, vshear_6000, shear_6000,
                                      temp_1000, temp_850, temp_700, temp_500, temp_300,
                                      rh_1000, rh_850, rh_700, rh_500, rh_300, vvel_1000,
                                      vvel_850, vvel_700, vvel_500, vvel_300))

        return input_hour"""

    """
    CREATING TRAINING DATA
    -------------------------
    Creates all the necessary data to train a model with. The output is stored under the
    variable 'target' and is saved in /training/X/; the input is stored under the
    variable 'data' and saved in /training/y/.
    
    The output created with this method is different than the one found in create_output().
    The output is mapped onto the grid of the model, with any grid containing any initiation
    will be marked with the value 1.
    
    Once all of the data has been saved into the disk, calling the method again will just
    load all of the saved data into memory into the object unless 'replace_old' is False.



    Attributes:
        days (list) - list of all the days (YYYY-MM-DD) to compute (12z to 12z next day)
        
        Other keyword arguments:
            hours - list of all the hours to compute from each day. default is all hours 
                    from 18z to 04z
            replace_old (bool) - if True, it replaces all data files that have already 
                                 been saved before
            log (bool) - whether or not to print text that logs the progress of the method
            


    Returns (X, y):
        X - numpy array of all the input data for a model
        y - numpy array of all the output to train a model
    """
    # days: YYYY-MM-DD
    def create_training(self, days: list, **kwargs):
        hours: list = kwargs.get('hours', ['18', '19', '20', '21', '22',
                                           '23', '0', '1', '2', '3', '4'])
        replace_old: bool = kwargs.get('replace_old', False)
        log: bool = kwargs.get('log', True)
        add_to_field = kwargs.get('add_to_field', True)

        X_list = []
        y_list = []

        # adding training data for every hour in every day
        for day in days:
            for hour in hours:

                mrms_day = day.split(' ')[0].replace('-', '')
                mrms_hour = int(hour)

                if log:
                    print(mrms_day)

                # converting the string date into YYYYMMDD for create_output()
                utc_day = day
                if int(hour) < 12:
                    mrms_datetime = datetime.strptime(mrms_day, '%Y%m%d') + timedelta(days=1)
                    utc_day = mrms_datetime.strftime('%Y-%m-%d')
                    mrms_day = mrms_datetime.strftime('%Y%m%d')

                X_filename = f"training/X/{mrms_day}-{mrms_hour}_X.npy"
                y_filename = f"training/y/{mrms_day}-{mrms_hour}_y.npy"

                # boolean variable on whether to compute the training data or not
                # computes if the data is already saved and it does not need to be replaced
                compute = True
                if os.path.exists(y_filename):
                    if log:
                        print(f"Output at {mrms_day} {mrms_hour}:00 already exists")
                    compute = False

                    if replace_old:
                        compute = True
                        if log:
                            print('Replacing...')

                # computing initiation
                if compute:
                    if log:
                        print(f'Computing initiation for {mrms_day} {mrms_hour}:00')

                    label = self.create_output(mrms_day, mrms_hour)
                    scaled_label = self.scale_output(label)

                    # saves the output data onto the disk
                    np.save(y_filename, scaled_label.flatten())

                # boolean variable on whether to compute the training data or not
                # computes if the data is already saved and it does not need to be replaced
                compute = True
                if os.path.exists(X_filename):
                    if log:
                        print(f"Input at {mrms_day} {mrms_hour}:00 already exists")
                    compute = False

                    if replace_old:
                        compute = True
                        if log:
                            print('Replacing...')

                # computing input data
                if compute:
                    if log:
                        print(f"Computing input for {mrms_day} {mrms_hour}:00")

                    M = Model(self.model, f'{mrms_day} {hour}:00', 0)
                    input_hour = self.create_input(M)

                    # saves the input data onto the disk
                    np.save(X_filename, input_hour)

                # appends all hours into a list
                self.days.append(f'{utc_day} {hour}:00')

                # loads the data into a list
                X_list.append(np.load(X_filename))
                y_list.append(np.load(y_filename))

                warnings.filterwarnings("ignore")

        # turns the list into a single numpy array
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # whether or not to append the new training data to the class fields
        if add_to_field:
            if self.data is None:
                self.data = X
            else:
                self.data = np.concatenate([self.data, X])

            if self.target is None:
                self.target = y
            else:
                self.target = np.concatenate([self.target, y])

        else:
            self.data = X
            self.target = y

        return X, y

    """
    ADDING FEATURES
    -------------------------
    Adds features to the current training data. Make sure to keep track of all of the 
    features loaded in different files and used in training the model so that there are 
    no discrepencies. If you want to use this, I recommend to add the feature to all 
    data files for no confusion (if save is True).

    Attributes:
        features: list - List of features to add. Must be available from data.json
        save (boolean) - Whether or not to save the new fields to the data files.
    
    Returns nothing
    """
    def add_features(self, features: list, save=True):

        # opening json file that stores each model field
        with open("data.json", "r") as f:
            data = json.load(f)
            field_list = data["herbie"] # all fields which can be directly extracted
            calculate_fields = data["calculate"] # fields which need to be calculated

        fields = [field for field in field_list] # all names of the fields in one concise list
        fields = fields + calculate_fields # list of every field available

        valid_fields = [] # fields which can be inputted
        for feature in features:
            # checking if the field is inside data.json
            if feature not in fields:
               raise NameError(f'{feature} is not available to use. Valid features are: \n{fields}')

            # checking if the field has already been loaded
            if feature in self.feature_names:
                print(f'{feature} is already loaded!')
            else:
                valid_fields.append(feature)

        # list of all new fields as flattened numpy arrays
        X_list = [] 

        if len(valid_fields) > 0:

            # adding the field to each hour in the training object
            for hour in self.days:

                # extracting time and date from the self.days string of days
                timedate = datetime.strptime(hour, '%Y-%m-%d %H:%M')
                timedate_hour = timedate.strftime("%Y%m%d")
                timedate_hour += "-" + str(timedate.hour)

                # opening the file with the numpy data
                filename = f'training/X/{timedate_hour}_X.npy'
                X_hour = np.load(filename)

                # getting fields from create_input()
                M = Model(self.model, hour, 0)
                new_features = self.create_input(M, features=valid_fields)

                # horizontally stacking (adding columns of fields) to the original dataset
                X = np.hstack((X_hour, new_features))
                X_list.append(X)

                # (OPTIONAL BUT RECOMMENDED) updates the training data files with the new fields
                if save:
                    np.save(filename, X)

            # updating the object's fields
            self.data = np.vstack(X_list)
            self.feature_names += valid_fields


    # init_hour = YYYY-MM-DD HH:MM
    def add_running(self, init_hour, run_hours, add_to_field):
        dt_init = datetime.strptime(init_hour, '%Y-%m-%d %H:%M')

        X_list = []
        for hour in range(run_hours + 1):
            run_hour = (dt_init + timedelta(hours=hour)).strftime('%Y-%m-%d %H:%M')

            M = Model(self.model, init_hour, run_hour)
            input_hour = self.create_input(M)

            X_list.append(input_hour)

        X = np.vstack(X_list)

        if add_to_field:
            if self.data is None:
                self.data = np.concatenate([self.data, X])

        else:
            self.data = X

        return X

    """
    EXTRACTING TRAINING DATA AT AN HOUR
    -------------------------
    Returns the numpy array of the output and input for a specific hour from the training 
    data. Print the variable 'days' to find the available hours
    
    Attributes:
        hour (string) - hour (YYYY-MM-DD HH:MM)  
    
    Returns (X, y):
        X - numpy array of all the input data at the hour
        y - numpy array of all the output at the hour
    """
    def extract_hour(self, hour):
        # finds the index of the hour inside the 'days' list
        hour_index = None
        for i, time in enumerate(self.days):
            if datetime.strptime(time, '%Y-%m-%d %H:%M') == datetime.strptime(hour, '%Y-%m-%d %H:%M'):
                hour_index = i

        if hour_index == None:
             raise NameError(f'Could not find hour. Make sure the hour is in YYYY-MM-DD HH:MM format.\nPrint Training.days to check the list of days')

        # indexes for the hour in 'target' and 'data' variables
        data_index = hour_index * self.grid_size
        end_index = (hour_index + 1) * self.grid_size

        X = self.data[data_index:end_index, :]
        y = self.target[data_index:end_index]

        return X, y

    """
    COMPOSITE TARGET
    -------------------------
    Creates a composite of all the convective initiation for several hours. The output
    must already be loaded into the object.

    Attributes:
        start_hour - first hour (YYYY-MM-DD HH:MM) of the composite
        hours (int) - how many hours in total for the composite

    Returns (target):
        target - 1D numpy array of the convective initiation"""
    def composite_target(self, start_hour, hours: int):
        start_hour_datetime = datetime.strptime(start_hour, '%Y-%m-%d %H:%M')

        # adds all the data for each hour to the list
        y_list = []
        for i in range(hours):
            hour = start_hour_datetime + timedelta(hours=i)
            hour = hour.strftime('%Y-%m-%d %H:%M')

            _, y = self.extract_hour(hour)
            y_list.append(y)

        # finds the maximum value between the list of the hours
        target = np.maximum.reduce(y_list)

        return target

    """
    RETURN FEATURE INDEX
    -------------------------
    Returns the number used to index a specific feature in the input.

    Attributes:
        feature - feature from 'feature_names'

    Returns (target):
        feature_index - index of the feature inside 'input'"""
    def return_feature_index(self, feature):
        try:
            feature_index = self.feature_names.index(feature)
        except NameError:
            print(f'{feature} is not a valid feature. These are the valid product names:\n{valid_features}')

        return feature_index

    """
    INITIALIZE THE TRAINING OBJECT
    -------------------------
    Creates a new 'Training' object based off a weather model, extent in the CONUS, and a list
    of days. All input and output data is saved for each hour under /training/X/ and training/y/
    in this directory as numpy files. 
    
    Each day spans from 12z in the morning to 12z in the next day, but is saved as the next day 
    for the hour in the variable 'days'



    Attributes:
        model - model to create the data off of ('rap')
        extent - list [min lat, max lat, min lon, max lon] to specify the extent in the CONUS to 
                 save the data as
        days - list (YYYY-MM-DD) 
        
        Other keyword arguments:
            hours - list of all the hours to compute from each day. default is all hours 
                    from 18z to 04z
            replace_old (bool) - if True, it replaces all data files that have already 
                                 been saved before
            log (bool) - whether or not to print text that logs the progress of the method"""
    def __init__(self, model: str, extent: list):
        self.model = model
        self.extent = extent
        (self.min_lat, self.max_lat,
         self.min_lon, self.max_lon) = extent

        reference = Model(model, '2021-05-02 00:00', 0)
        reference_grid = reference.retrieve_field(['refc'])['refc']

        self.min_x, self.min_y = reference.return_coords(self.min_lat, self.min_lon, dataset=reference_grid)
        self.max_x, self.max_y = reference.return_coords(self.max_lat, self.max_lon, dataset=reference_grid)

        self.reference_grid = reference_grid.sel(
            x=slice(self.min_x, self.max_x),
            y=slice(self.min_y, self.max_y)
        )

        self.longitude = self.reference_grid.longitude.values
        self.latitude = self.reference_grid.latitude.values
        self.grid_shape = self.longitude.shape
        self.grid_size = self.longitude.shape[0] * self.longitude.shape[1]
        self.x = self.reference_grid.x.values
        self.y = self.reference_grid.y.values

        self.feature_names = ["sbcape", "sbcin", "mlcape", "mlcin", "mucape", "mucin",
                         "2m temp", "2m dewp", "lfc", "pbl", "li", "dew_dep", "sfc_conv", "u",
                         "v", "wspd", "u_500", "v_500", "wspd_500", "u_300", "v_300",
                         "wspd_300", "u_700", "v_700", "wspd_700", "u_850", "v_850",
                         "wspd_850", "ushear_6000", "vshear_6000", "shear_6000",
                         "temp_1000", "temp_850", "temp_700", "temp_500", "temp_300",
                         "rh_1000", "rh_850", "rh_700", "rh_500", "rh_300", "vvel_1000",
                         "vvel_850", "vvel_700", "vvel_500", "vvel_300"]

        self.days = []

        self.data = None
        self.target = None
