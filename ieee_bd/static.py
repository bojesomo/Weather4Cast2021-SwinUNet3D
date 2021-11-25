import netCDF4
import os
import numpy as np


region_coordinates = {
    "R3": (935, 400),
    "R6": (1270, 250),
    "R2": (1550, 200),
    "R1": (1850, 760),
    "R5": (1300, 550),
    "R4": (1020, 670),
    "R7": (1700, 470),
    "R8": (750, 670),
    "R9": (450, 760),
    "R10": (250, 500),
    "R11": (1000, 130),
}
region_size = (256, 256)


class StaticData:
    def __init__(self, data_dir=None,
                 regions=None):
        if data_dir is None:
            data_dir = os.path.join(
                '/home/farhanakram/Alabi/Datasets/Weather4cast2021/',
                'statics'
            )

        fn_latlon = os.path.join(data_dir,
                                 "Navigation_of_S_NWC_CT_MSG4_Europe-VISIR_20201106T120000Z.nc")
        with netCDF4.Dataset(fn_latlon, 'r') as ds:
            self.lon = np.array(ds["longitude"][0, :, :]).astype(np.float32)
            self.lat = np.array(ds["latitude"][0, :, :]).astype(np.float32)
        self.lon = (self.lon + 76) / (76 + 76)
        self.lat = (self.lat - 23) / (86 - 23)

        fn_elev = os.path.join(data_dir,
                               "S_NWC_TOPO_MSG4_+000.0_Europe-VISIR.raw")
        self.elevation = np.fromfile(fn_elev, dtype=np.float32).reshape(self.lon.shape)
        self.elevation[self.elevation < 0] = 0
        self.elevation /= self.elevation.max()

    def get_data(self, region, variables):  # , box):
        (reg_j0, reg_i0) = region_coordinates[region]
        reg_i1 = reg_i0 + region_size[0]
        reg_j1 = reg_j0 + region_size[1]
        # ((i0, i1), (j0, j1)) = box
        var = {
            "latitude": self.lat,
            "longitude": self.lon,
            "elevation": self.elevation
        }
        # data = [var[v][reg_i0:reg_i1, reg_j0:reg_j1][i0:i1, j0:j1] for v in variables]
        data = [var[v][reg_i0:reg_i1, reg_j0:reg_j1] for v in variables]
        return np.stack(data, axis=0)
