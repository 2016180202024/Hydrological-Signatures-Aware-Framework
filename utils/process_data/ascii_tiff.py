import numpy as np
import xarray as xr
import rioxarray


# 将.asc文件转为.tif文件
def ascii_to_tiff(asc_path: str, tif_path: str, tif_attrs: dict = {}) -> None:
    """ASCII 文件转 TIFF 文件
    :param asc_path: ASCII 文件路径, 例如: ./test.asc
    :type asc_path: str
    :param tif_path: TIFF 文件输出路径, 例如: ./test.tif
    :type tif_path: str
    :param tif_attrs: TIFF 文件属性, 例如: {"unit": "m"}, defaults to {}
    :type tif_attrs: dict, optional
    """
    # 获取 ASCII 文件前 6 行属性值
    attrs: dict = {}
    with open(asc_path, "r") as file:
        for _ in range(6):
            line: str = file.readline().strip().split(" ")
            attrs[line[0].lower()] = eval(line[-1])
    if "xllcenter" not in attrs.keys():
        attrs["xllcenter"] = attrs["xllcorner"] + 0.5 * attrs["cellsize"]
        attrs["yllcenter"] = attrs["yllcorner"] + 0.5 * attrs["cellsize"]

    # 计算每个点经纬度坐标
    longitudes = [attrs["xllcenter"] + i * attrs["cellsize"] for i in range(attrs["ncols"])]
    latitudes = [attrs["yllcenter"] + i * attrs["cellsize"] for i in range(attrs["nrows"])]
    latitudes.reverse()

    # 读取 ASCII 文件矩阵数值
    data = np.loadtxt(asc_path, skiprows=6)
    data[data == attrs["nodata_value"]] = np.nan
    da = xr.DataArray(data, coords=[latitudes, longitudes], dims=["y", "x"])
    # 设置 TIFF 文件属性值
    tif_attrs["NODATA_VALUE"] = attrs["nodata_value"]
    da.attrs = tif_attrs
    # 设置 TIFF 文件参考系信息
    rioxarray.raster_array.RasterArray(da)
    da.rio.write_crs("epsg:4326", inplace=True)
    da.rio.to_raster(tif_path)


def tiff_to_ascii(tif_path: str, asc_path: str) -> None:
    """TIFF 文件转 ASCII 文件
    :param tif_path: TIFF 文件路径, 例如: ./test.tif
    :type tif_path: str
    :param asc_path: ASCII 输出文件路径, 例如: ./test.asc
    :type asc_path: str
    """
    # 读取 TIFF 文件
    tif = rioxarray.open_rasterio(tif_path)
    shape = tif.rio.shape
    transform = tif.rio.transform()

    # 获取 ASCII 文件前 6 行属性
    attrs: dict = {}
    attrs["ncols"] = shape[1]
    attrs["nrows"] = shape[0]
    attrs["xllcorner"] = transform[2]
    attrs["yllcorner"] = transform[5] + shape[0] * transform[4]
    attrs["cellsize"] = transform[0]
    attrs["nodata_value"] = tif.rio.nodata if tif.rio.nodata else -9999

    # 获取数据
    data = tif.values[0]
    data[np.isnan(data)] = attrs["nodata_value"]

    # 写入文件
    with open(asc_path, "w") as file:
        for key, value in attrs.items():
            file.write(f"{key.upper():14}{value}\n")
        np.savetxt(fname=file, X=data, fmt="%.2f")


if __name__ == "__main__":
    ascii_to_tiff(r"D:\data\Road Density\grip4_total_dens_m_km2.asc",
                  "D:\data\Road Density\grip4_total_dens_m_km2.tif",
                  {"UNIT": "m"})
    # tiff_to_ascii("./tifs/dem.tif", "./ascs/dem2.asc")