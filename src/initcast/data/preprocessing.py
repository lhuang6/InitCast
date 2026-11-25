from herbie import Herbie
import json
import metpy.calc as mpcalc
from metpy.units import units

with open("data.json", "r") as f:
    data = json.load(f)
    input_features = data["herbie"]
    calculate_features = data["calculate"]

def get_input_features(list):
    temp_input_features = {}
    for arg in list:
        if arg in input_features:
            temp_input_features[arg] = input_features[arg]

    if list == None:
        return input_features
    else:
        return temp_input_features

"""NEEDS WORK





"""

def calculate(name, feature_list, dx=13000, dy=13000):
    if name not in calculate_features:
        return ValueError(f"{name} is not a valid feature to calculate! The valid features are: {calculate_features}")

    dx = dx * units('m')
    dy = dy * units('m')

    u_name = "u"
    v_name = "v"
    units = ''

    if "_" in name:
        level = name.split("_")[-1]
        u_name = f"u_{level}"
        v_name = f"v_{level}"

    vort_name = f"vort_{level}"

    if u_name not in feature_list and v_name not in feature_list:
        feature_list = get_herbie(H, [u_name, v_name])

    u = feature_list[u_name]["data"]
    v = feature_list[v_name]["data"]
    
    if "wspd" in name or name == "shear_6000":
        parameter = (u ** 2 + v ** 2) ** 0.5
        units = "m/s"

    if name == "sfc_conv":
        parameter = mpcalc.divergence(u, v, dx=dx, dy=dy)
        units = "1/s"

    if "vort_" in name:
        parameter = mpcalc.vorticity(u, v, dx=dx, dy=dy)
        units = "1/s"

    if "va_" in name:
        if vort_name in feature_list:
            vort = feature_list[vort_name]["data"]
        else:
            vort = mpcalc.vorticity(u, v, dx=dx, dy=dy)
        
        parameter = mpcalc.advection(vort, u, v, dx=dx, dy=dy)
        units = "1/s^2"

    if "dpva" in name:
        va_list = []
        for va_name in "va_500", "va_300":
            if va_name in feature_list:
                va = feature_list[va_name]["data"]
            else:
                vort_name = va_name.replace("va", "vort")
                if vort_name in feature_list:
                    vort = feature_list[vort_name]["data"]
                else:
                    u_name = va_name.replace("va", "u")
                    v_name = va_name.replace("va", "v")
                    if u_name in feature_list and v_name in feature_list:
                        u = feature_list[u_name]["data"]
                        v = feature_list[v_name]["data"]
                    else:
                        uv = get_herbie(H, [u_name, v_name])
                        u = uv[u_name]["data"]
                        v = uv[v_name]["data"]

                    vort = mpcalc.vorticity(u, v, dx=dx, dy=dy) 
                va = mpcalc.advection(vort, u, v, dx=dx, dy=dy)

            va_list.append(va)

        parameter = va_list[0] - va_list[1]
        units = "1/s^2"

    if "advection" in name or "q_vect_conv" in name:
        if "t" not in feature_list:
            t = get_herbie(H, ["t"])["t"]["data"]
        else:
            t = feature_list["t"]["data"]

    if "advection_" in name:
        t = data[2] * units('K')
        parameter = mpcalc.advection(t, u, v, dx=dx, dy=dy)
        units = "K/s"

    if name == "q_vect_conv":
        t = data[2] * units('K')
        q_vect = mpcalc.q_vect(u, v, t, 850 * units.hPa, dx=dx, dy=dy)
        parameter = mpcalc.diverence(q_vect[0], q_vect[1], dx=dx, dy=dy)

        units = 'm/(kg s)'

    ds = parameter.to_dataset(name)

    feature = { name: {
        "data": ds,
        "xarray_name": name,
        "units": units
    }
    }

    return feature

def get_herbie(H: Herbie, list):
    if not args:
        args = input_features

    temp_feature_list = {}
    temp_calc_list = []
    for feature in list:
        if feature in input_features:
            xarray = H.xarray(rf"{input_features[feature]["search"]}")

            temp_feature_list[feature] = {
                "data": xarray,
                "xarray_name": input_features[feature]["xarray_name"],
                "units": input_features[feature]["units"]
            }

        elif feature in calculate_features:
            temp_calc_list.append(feature)

        else:
            raise ValueError(f"{feature} is not a valid feature! The valid features are: {list(input_features.keys()) + calculate_features}")

    for calc in temp_calc_list:

        ordered_calc = sorted(calculate_features, key=lambda x: calculate_features.index(x))

        for feature in ordered_calc:
            calc_feature = calculate(feature, temp_feature_list)

        temp_feature_list.update(calc_feature)

    return temp_feature_list
    