import glob
import os

import pandas as pd
from osgeo import gdal, ogr, osr
import geopandas as gpd


# 将shp的图形转为代表图形的几何中心点
def area_to_point(in_path, out_path):
    """
    矢量裁剪
    :param in_path: 输入矢量图形文件
    :param out_path: 输出矢量点状文件
    :return:
    """
    # 解决中文字符问题
    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")
    # 注册所有的驱动
    ogr.RegisterAll()
    ds = ogr.Open(in_path, 0)
    if ds is None:  # 打开失败
        print("打开失败")
    # 获取该数据源中的图层个数，一般shp数据图层只有一个，如果是mdb、dxf等图层就会有多个
    layer_count = ds.GetLayerCount()
    if layer_count != 1:
        print("图层数量异常")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = driver.CreateDataSource(out_path)
    in_lyr = ds.GetLayer()
    out_lyr = out_ds.CreateLayer('Point', in_lyr.GetSpatialRef(), geom_type=ogr.wkbPoint)
    # 获取文件字段属性
    in_feature = in_lyr.GetNextFeature()
    # 复制全部的字段
    in_feature_defn = in_lyr.GetLayerDefn()
    field_names = []
    for i in range(0, in_feature_defn.GetFieldCount()):
        field_defn = in_feature_defn.GetFieldDefn(i)
        field_name = field_defn.name
        field_names.append(field_name)
        # 输出layer创建字段
        out_lyr.CreateField(field_defn)
    # 获取输出图层属性表信息
    out_feature_defn = out_lyr.GetLayerDefn()
    feature = in_lyr.GetFeature(0)
    # 遍历要素
    while feature:
        new_feature = ogr.Feature(out_feature_defn)
        # 添加点要素
        geom = feature.GetGeometryRef()
        center = geom.Centroid()
        new_feature.SetGeometry(center)
        # 复制所有字段属性
        for field_name in field_names:
            area = feature.GetField(field_name)
            # 添加点的字段值
            new_feature.SetField(field_name, area)
        # 添加要素到图层
        out_lyr.CreateFeature(new_feature)
        feature = in_lyr.GetNextFeature()
    out_ds.Destroy()


# 使用shp掩膜shp文件，掩膜shp需要比被掩膜shp大
def shp_mask_shp(baseFilePath,
                 maskFilePath,
                 saveFolderPath):
    """
	:param baseFilePath: 要裁剪的矢量文件
	:param maskFilePath: 掩膜矢量文件
	:param saveFolderPath: 裁剪后的矢量文件保存目录
	"""
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 载入要裁剪的矢量文件
    baseData = ogr.Open(baseFilePath)
    baseLayer = baseData.GetLayer()
    spatial = baseLayer.GetSpatialRef()
    geomType = baseLayer.GetGeomType()
    baseLayerName = baseLayer.GetName()
    # 载入掩膜矢量文件
    maskData = ogr.Open(maskFilePath)
    maskLayer = maskData.GetLayer()
    maskLayerName = maskLayer.GetName()
    # 生成裁剪后的矢量文件
    outLayerName = maskLayerName + "_mask_" + baseLayerName
    outFileName = outLayerName + ".shp"
    outFilePath = os.path.join(saveFolderPath, outFileName)
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    outData = driver.CreateDataSource(outFilePath)
    outLayer = outData.CreateLayer(outLayerName, spatial, geomType)
    baseLayer.Clip(maskLayer, outLayer)
    outData.Release()
    baseData.Release()
    maskData.Release()


def concat_shp(input_files, output_file):
    # 创建输出文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    output_ds = driver.CreateDataSource(output_file)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # 定义输出的坐标系为"WGS 84"
    output_layer = output_ds.CreateLayer("combined", srs=srs, geom_type=ogr.wkbPolygon)
    # 遍历每个输入文件
    for input_file in input_files:
        input_ds = ogr.Open(input_file)
        input_Layer = input_ds.GetLayer()
        # 获取输入图是中的要素
        for feature in input_Layer:
            # 创建新要素
            output_feature = ogr.Feature(output_layer.GetLayerDefn())
            output_feature.SetGeometry(feature.GetGeometryRef())
            # 将哥要素添加到输出图号
            output_layer.CreateFeature(output_feature)
            # 关闲输入巅据源
            input_ds = None
        # 保存并关闲输出巅据源
    output_ds = None
    print("矢量文件已成功合并为", output_file)
