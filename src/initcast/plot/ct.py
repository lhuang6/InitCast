from matplotlib.colors import ListedColormap
import numpy as np

def black_ct():
    black = np.ones((2, 4))
    black[:, 0] = np.linspace(0, 0, 2)
    black[:, 1] = np.linspace(0, 0, 2)
    black[:, 2] = np.linspace(0, 0, 2)
    blackct = ListedColormap(black)

    return blackct

def dwpt_ct():
    dewpoint = np.ones((130, 4))
    dewpoint[0:40, 0] = np.linspace(152 / 255, 73 / 255, 40)
    dewpoint[0:40, 1] = np.linspace(109 / 255, 65 / 255, 40)
    dewpoint[0:40, 2] = np.linspace(77 / 255, 54 / 255, 40)

    dewpoint[40:80, 0] = np.linspace(73 / 255, 244 / 255, 40)
    dewpoint[40:80, 1] = np.linspace(65 / 255, 242 / 255, 40)
    dewpoint[40:80, 2] = np.linspace(54 / 255, 215 / 255, 40)

    dewpoint[80:90, 0] = np.linspace(230 / 255, 95 / 255, 10)
    dewpoint[80:90, 1] = np.linspace(245 / 255, 196 / 255, 10)
    dewpoint[80:90, 2] = np.linspace(230 / 255, 95 / 255, 10)

    dewpoint[90:100, 0] = np.linspace(48 / 255, 8 / 255, 10)
    dewpoint[90:100, 1] = np.linspace(174 / 255, 78 / 255, 10)
    dewpoint[90:100, 2] = np.linspace(48 / 255, 8 / 255, 10)

    dewpoint[100:110, 0] = np.linspace(97 / 255, 19 / 255, 10)
    dewpoint[100:110, 1] = np.linspace(163 / 255, 44 / 255, 10)
    dewpoint[100:110, 2] = np.linspace(175 / 255, 43 / 255, 10)

    dewpoint[110:120, 0] = np.linspace(102 / 255, 45 / 255, 10)
    dewpoint[110:120, 1] = np.linspace(102 / 255, 30 / 255, 10)
    dewpoint[110:120, 2] = np.linspace(154 / 255, 100 / 255, 10)

    dewpoint[120:130, 0] = np.linspace(114 / 255, 164 / 255, 10)
    dewpoint[120:130, 1] = np.linspace(63 / 255, 113 / 255, 10)
    dewpoint[120:130, 2] = np.linspace(113 / 255, 129 / 255, 10)

    dewpointct = ListedColormap(dewpoint)

    return dewpointct

def piv_wspd_ct():
    piv_wspd = np.ones((120, 4))
    piv_wspd[0:10, 0] = np.linspace(1, 102/256, 10)
    piv_wspd[0:10, 1] = np.linspace(1, 208/256, 10)
    piv_wspd[0:10, 2] = np.linspace(1, 251/256, 10)

    piv_wspd[10:20, 0] = np.linspace(102 / 256, 107/256, 10)
    piv_wspd[10:20, 1] = np.linspace(208 / 256, 90/256, 10)
    piv_wspd[10:20, 2] = np.linspace(251 / 256, 204/256, 10)

    piv_wspd[20:30, 0] = np.linspace(107 / 256, 245/256, 10)
    piv_wspd[20:30, 1] = np.linspace(90 / 256, 146/256, 10)
    piv_wspd[20:30, 2] = np.linspace(204 / 256, 217/256, 10)

    piv_wspd[30:50, 0] = np.linspace(245 / 256, 177/256, 20)
    piv_wspd[30:50, 1] = np.linspace(146 / 256, 0, 20)
    piv_wspd[30:50, 2] = np.linspace(217 / 256, 146/256, 20)

    piv_wspd[50:60, 0] = np.linspace(177 / 256, 223/256, 10)
    piv_wspd[50:60, 1] = np.linspace(0, 0, 10)
    piv_wspd[50:60, 2] = np.linspace(146 / 256, 0, 10)

    piv_wspd[60:70, 0] = np.linspace(223 / 256, 244/256, 10)
    piv_wspd[60:70, 1] = np.linspace(0, 0, 10)
    piv_wspd[60:70, 2] = np.linspace(0, 40/256, 10)

    piv_wspd[70:80, 0] = np.linspace(244 / 256, 1, 10)
    piv_wspd[70:80, 1] = np.linspace(0, 59/256, 10)
    piv_wspd[70:80, 2] = np.linspace(40 / 256, 63/256, 10)

    piv_wspd[80:90, 0] = np.linspace(1, 1, 10)
    piv_wspd[80:90, 1] = np.linspace(59 / 256, 240/256, 10)
    piv_wspd[80:90, 2] = np.linspace(63 / 256, 103/256, 10)

    piv_wspd[90:120, 0] = np.linspace(1, 174/256, 30)
    piv_wspd[90:120, 1] = np.linspace(240 / 256, 86/256, 30)
    piv_wspd[90:120, 2] = np.linspace(103 / 256, 0, 30)

    piv_wspdct = ListedColormap(piv_wspd)

    return piv_wspdct

def composite_ct(type=None):

    composite = np.ones((100, 4))

    composite[0:10, 0] = np.linspace(1, 105 / 255, 10) # gray
    composite[0:10, 1] = np.linspace(1, 105 / 255, 10)
    composite[0:10, 2] = np.linspace(1, 105 / 255, 10)

    composite[10:20, 0] = np.linspace(55 / 255, 167 / 255, 10) # blue
    composite[10:20, 1] = np.linspace(83 / 255, 200 / 255, 10)
    composite[10:20, 2] = np.linspace(106 / 255, 206 / 255, 10)

    composite[20:30, 0] = np.linspace(233 / 255, 225 / 255, 10) # yellow/orange
    composite[20:30, 1] = np.linspace(221 / 255, 111 / 255, 10)
    composite[20:30, 2] = np.linspace(150 / 255, 2 / 255, 10)

    composite[30:40, 0] = np.linspace(220 / 255, 139 / 255, 10) # red
    composite[30:40, 1] = np.linspace(65 / 255, 9 / 255, 10)
    composite[30:40, 2] = np.linspace(16 / 255, 80 / 255, 10)

    composite[40:50, 0] = np.linspace(115 / 255, 218 / 255, 10) # purple
    composite[40:50, 1] = np.linspace(8 / 255, 153 / 255, 10)
    composite[40:50, 2] = np.linspace(138 / 255, 231 / 255, 10)

    composite[50:60, 0] = np.linspace(233 / 255, 178 / 255, 10)  # pink
    composite[50:60, 1] = np.linspace(190 / 255, 68 / 255, 10)
    composite[50:60, 2] = np.linspace(195 / 255, 90 / 255, 10)

    composite[60:100, 0] = np.linspace(137 / 255, 201 / 255, 40)  # brown
    composite[60:100, 1] = np.linspace(61 / 255, 166 / 255, 40)
    composite[60:100, 2] = np.linspace(72 / 255, 168 / 255, 40)

    if type == "cin":
        composite = composite[::-1]

    compositect = ListedColormap(composite)

    return compositect
