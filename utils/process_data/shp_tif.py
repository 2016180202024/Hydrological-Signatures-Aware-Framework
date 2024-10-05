from pathlib import Path

from osgeo import gdal, ogr


def Vector_To_Raster(input_shp, refer_raster, attribute, output_raster):
    """
    :param input_shp: 输入需要转换的矢量数据
    :param refer_raster: 输入参考栅格数据
    :param attribute: 输入栅格值对应的矢量字段
    :param output_raster: 输出栅格数据路径
    :return: None
    """
    ds_raster = gdal.Open(refer_raster)
    ds_proj = ds_raster.GetProjection()  # 投影信息
    ds_trans = ds_raster.GetGeoTransform()  # 仿射地理变换参数
    ds_width = ds_raster.RasterXSize  # 获取宽度/列数
    ds_height = ds_raster.RasterYSize  # 获取高度/行数
    ds_raster = None  # 释放内存
    del ds_raster
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(input_shp)
    layer = ds.GetLayer()  # 获取图层文件对象
    result = gdal.GetDriverByName('GTiff').Create(output_raster, ds_width, ds_height, bands=1, eType=gdal.GDT_Byte)
    result.SetGeoTransform(ds_trans)  # 写入仿射地理变换参数
    result.SetProjection(ds_proj)  # 写入投影信息
    band = result.GetRasterBand(1)
    band.SetNoDataValue(0)  # 忽略背景值
    band.FlushCache()  # 清空数据缓存
    # options = ["ATTRIBUTE=attribute", "CHUNKY SIZE=0", "ALL_TOUCHED=False"]
    # 指定输入矢量数据的属性字段中的字段值作为栅格值写入栅格文件中，该值将输出到所有输出波段中。假如该值指定了，burn_Values参数的值将失效数可以设置为空。
    # 指定该运行操作的块的高度。该值越大所需的计算时间越小。如果该值没有设置或者设置为0则由GDAL的缓存大小根据公式：缓存所占的字节数/扫描函数的字节数得到。所以该值不会超出缓存的大小。
    # 设置为TRUE表示所有的像素接触到矢量中线或多边形，否则只是多边形中心或被Bresenham算法选中的部分。默认值为FALSE。简单来说，FALSE是八方向栅格化，TRUE是全路径栅格化。
    gdal.RasterizeLayer(result, [1], layer, burn_values=[1], options=[f"ATTRIBUTE={attribute}"])
    # 矢量转栅格函数。输出栅格，波段数，图层，栅格数值（可控失效），可选参数（栅格数值取字段）
    result = None
    del result, layer


Vector_To_Raster(r'D:\data\dem.shp',
                 r'D:\data\meteorology\climate\0.25\p_mean.tif',
                 'class_num',
                 r'D:\data\Geology\GLiM\glim.tif')
