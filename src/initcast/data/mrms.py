"""
All the methods used to download the MRMS radar data


Methods:
    extract_dt(key)
        returns a datetime object from an MRMS file

    download_data(day, hour, data, return_files=False, between_hour=False, skip=1)
        downloads all the files between or after an hour

    plot(ds, type, extent):
        plots the datq
"""

import s3fs
import gzip
import shutil
import os
from pathlib import Path
import re
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.plots import ctables

"""
EXTRACT DATETIME
-------------------------
Returns a datetime object from the MRMS filename.


Attributes:
    key - filename
    
Returns datetime object for the filename
"""
def extract_dt(key):
    # Pull the timestamp out of an MRMS S3 key like '..._{YYYYMMDD}-{HHMMSS}.grib2.gz'
    m = re.search(r'(\d{8})-(\d{6})', key)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')


"""
DOWNLOADING MRMS DATA
-------------------------
Downloads all the files needed at a certain day and hour. All the files are downloaded into
this directory at /MRMS/ and split into folders by the type of data being downloaded and the 
day.


Attributes:
    day - day to download (YYYYMMDD)
    hour (string) - hour to download (HH)
    data - what type of radar feature to download
        'cref_max' - max composite reflectivity swath
        'cref' - composite reflectivity
        'echotops' - echotops
    return_files (bool) - whether to return a list of the path to all the files
    between_hour (bool) - True downloads all files +-30 mins of the hour, and False is during 
                          the hour
    skip (int) - how many files to skip downloading
    


Returns:
    grib_files (list) - list of the paths to all the files downloaded
"""
# between_hour downloads from 30 mins before to 30 mins after hour
def download_data(day, hour, data, return_files=False, between_hour=False, skip=1):

    s3 = s3fs.S3FileSystem(anon=True)

    if data == 'cref_max':
        directory = 'noaa-mrms-pds/CONUS/CREF_1HR_MAX_00.50/'
        folder_base = 'MRMS/cref_max/'
    elif data == 'cref':
        directory = 'noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/'
        folder_base = 'MRMS/cref/'
    elif data == 'echotops':
        directory = 'noaa-mrms-pds/CONUS/EchoTop_18_00.50/'
        folder_base = 'MRMS/echotops/'
    else:
        raise ValueError(f"Unknown data type: {data}")

    center_time = datetime.strptime(day, '%Y%m%d') + timedelta(hours=int(hour))

    if between_hour:
        window_start = center_time - timedelta(minutes=30)
        window_end = center_time + timedelta(minutes=30)
    else:
        window_start = center_time
        window_end = center_time + timedelta(hours=1)

    # search a little wider than the window so we have candidates available
    # just past window_end (needed to find the "closes out the hour" file,
    # which might land a few minutes late, e.g. 23:04 instead of 23:00)
    search_start = window_start - timedelta(hours=1)
    search_end = window_end + timedelta(hours=1)

    days_needed = sorted({
        search_start.strftime('%Y%m%d'),
        window_start.strftime('%Y%m%d'),
        window_end.strftime('%Y%m%d'),
        search_end.strftime('%Y%m%d'),
    })

    all_files = []
    for d in days_needed:
        try:
            all_files.extend(s3.find(directory + d))
        except FileNotFoundError:
            continue

    # pair each file with its parsed timestamp, sorted chronologically
    dated_files = []
    for f in all_files:
        dt = extract_dt(f)
        if dt is not None:
            dated_files.append((dt, f))
    dated_files.sort(key=lambda x: x[0])

    # main selection: everything inside the window
    selected = []
    window_index = 0
    for i in range(len(dated_files)):
        dt, key = dated_files[i]

        if window_start <= dt <= window_end:
            if window_index % skip == 0:
                selected.append((dt, key))

            window_index += 1

    # selected = [(dt, key) for dt, key in dated_files if window_start <= dt <= window_end]

    # grab the single earliest file AFTER the window, regardless of how
    # many minutes late it is, so the hour is "closed out"
    after = [(dt, key) for dt, key in dated_files if dt > window_end]
    if after:
        selected.append(after[0])

    gz_files = []
    grib_files = []
    folder_files_cache = {}

    for dt, key in selected:
        file_day = dt.strftime('%Y%m%d')
        folder = f'{folder_base}{file_day}/'
        os.makedirs(folder, exist_ok=True)

        if file_day not in folder_files_cache:
            folder_files_cache[file_day] = [f for f in Path(folder).iterdir() if f.is_file()]

        gz_file = key.split('/')[-1]
        file_name = folder + gz_file
        grib_name = file_name.replace('.gz', '')
        grib_files.append(grib_name)

        if Path(grib_name) not in folder_files_cache[file_day]:
            print('downloading ' + gz_file + '...')
            s3.get(key, file_name)
            gz_files.append(file_name)

    for file in gz_files:
        grib_file = file.replace('.gz', '')
        with gzip.open(file, 'rb') as f_in:
            with open(grib_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file)

    if len(grib_files) == 1:
        grib_files = grib_files[0]

    if return_files:
        return grib_files

    return None


"""
PLOTTING MRMS DATA
-------------------------
Plots MRMS data. Only composite reflectivity is supported.


Attributes:
    ds - dataset to plot
    type - what the dataset is
        'cref' - composite reflectivity
    extent - list [min lat, max lat, min lon, max lon] that states the extent of the plot
"""
def plot(ds, type, extent):

    fig = plt.Figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    if type == 'cref':
        norm, cmap = ctables.registry.get_with_range('NWSReflectivity', 0, 75)
        cmap.set_under('white')

        plot = ax.pcolormesh(ds.longitude,
                             ds.latitude,
                             ds.unknown,
                             transform=ccrs.PlateCarree(),
                             cmap=cmap,
                             norm=norm)

    ax.set_extent(extent)
    plt.colorbar(plot, ax=ax)
