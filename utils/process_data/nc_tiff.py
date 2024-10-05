import os
import netCDF4 as nc
import numpy as np
from osgeo import gdal, osr, ogr
import glob


def nc_to_tif(nc_path, variable, output_dir_path):
    pre_data = nc.Dataset(nc_path)  # 利用.Dataset()读取nc数据
    lat_data = pre_data.variables['lat'][:]
    lon_data = pre_data.variables['lon'][:]
    pre_arr = np.asarray(pre_data.variables[variable])  # 属性变量名
    # 影像的左上角&右下角坐标
    Lonmin, Latmax, Lonmax, Latmin = [lon_data.min(), lat_data.max(),
                                      lon_data.max(), lat_data.min()]
    # 分辨率计算
    Num_lat = len(lat_data)
    Num_lon = len(lon_data)
    Lat_res = (Latmax - Latmin) / (float(Num_lat) - 1)
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon) - 1)
    # 创建tif文件
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = os.path.join(output_dir_path, variable + '.tif')
    out_tif = driver.Create(out_tif_name, Num_lon, Num_lat, 1, gdal.GDT_Int16)
    # 设置影像的显示范围
    geotransform = (Lonmin, Lon_res, 0.0, Latmax, 0.0, -Lat_res)
    out_tif.SetGeoTransform(geotransform)
    # 定义投影 WGS-84
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    # 数据导出
    out_tif.GetRasterBand(1).WriteArray(pre_arr)  # 将数据写入内存
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件


def write_tiff(outname, data, nl, ns, Lon_Res, Lat_Res, LonMin, LatMax, LonMax, LatMin):
    # 创建.tif文件
    driver = gdal.GetDriverByName("GTiff")
    out_tif = driver.Create(outname, ns, nl, 1, gdal.GDT_Float32)
    geotransform = (LonMin, Lon_Res, LatMin, LatMax, LonMax, -Lat_Res)
    out_tif.SetGeoTransform(geotransform)
    # 获取地理坐标系统信息，用于选取需要的地理坐标系统
    srs = osr.SpatialReference()
    proj_type = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    out_tif.SetProjection(proj_type)  # 给新建图层赋予投影信息
    # 数据写出
    out_tif.GetRasterBand(1).WriteArray(data)  # 将数据写入内存，此时没有写入硬盘
    out_tif.FlushCache()  # 将数据写入硬盘
    out_tif = None  # 注意必须关闭tif文件
