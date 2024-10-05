from pathlib import Path

from osgeo import gdal


def resample_images(path_refer, path_resample, out_path_resample):  # 影像重采样
    print("正在进行栅格重采样。。。")
    """
    :param path_refer: 重采样参考文件路径
    :param path_resample: 需要重采样的文件路径
    :param out_path_resample: 重采样后的输出路径
    """
    # 读取resample数据集
    ds_refer = gdal.Open(path_refer, gdal.GA_ReadOnly)  # 打开数据集dataset
    proj_refer = ds_refer.GetProjection()  # 获取投影信息
    trans_refer = ds_refer.GetGeoTransform()  # 获取仿射地理变换参数
    band_refer = ds_refer.GetRasterBand(1)    # 获取波段
    width_refer = ds_refer.RasterXSize  # 获取数据宽度
    height_refer = ds_refer.RasterYSize  # 获取数据高度
    bands_refer = ds_refer.RasterCount  # 获取波段数
    # 打开待采样数据集
    ds_resample = gdal.Open(path_resample, gdal.GA_ReadOnly)  # 打开数据集dataset
    proj_resample = ds_resample.GetProjection()  # 获取输入影像的投影信息
    nodata_resample = ds_resample.GetRasterBand(1).GetNoDataValue()  # nodata值
    # 创建输出数据集
    driver = gdal.GetDriverByName('GTiff')  # 定义输出的数据资源
    ds_output = driver.Create(out_path_resample, width_refer, height_refer, bands_refer, band_refer.DataType)  # 创建重采样影像
    ds_output.SetGeoTransform(trans_refer)  # 设置重采样影像的仿射地理变换
    ds_output.SetProjection(proj_refer)  # 设置重采样影像的投影信息
    ds_output.GetRasterBand(1).SetNoDataValue(nodata_resample)
    # 重采样
    # 输入数据集、输出数据集、输入投影、参考投影、重采样方法(最邻近内插\双线性内插\三次卷积等)、回调函数
    gdal.ReprojectImage(ds_resample, ds_output, proj_resample, proj_refer, gdal.GRA_Bilinear, 0.0, 0.0,)


def mosaic_tif(img_dir_path, out_file_path):
    """
    :param img_dir_path: 需要镶嵌的影像文件夹
    :param out_file_path: 镶嵌后输出的影像路径

    :return: None
    """
    img_dir_path = Path(img_dir_path)
    img_list = []
    input_proj = None
    for file_path in img_dir_path.glob('*.tif'):
        image = gdal.Open(file_path, gdal.GA_ReadOnly)
        img_list.append(image)
        if input_proj is None:
            input_proj = image.GetProjection()

    options = gdal.WarpOptions(srcSRS=input_proj, dstSRS=input_proj, format='GTiff',
                               resampleAlg=gdal.GRA_Bilinear)
    # 输入投影，输出投影，输出格式，重采样方法
    gdal.Warp(out_file_path, img_list, options=options)
    del img_list


def resample_tif(in_path, out_path, proj='EPSG:4326', xRes=0.25, yRes=0.25):
    # 重投影和重采样到proj
    gdal.Warp(out_path, in_path, dstSRS=proj,
              xRes=xRes, yRes=yRes, targetAlignedPixels=True)


if __name__ == '__main__':
    # vector2raster(inputfilePath=r'D:\data\Geology\GLHYMPS 2.0\GLHYMPS.shp',
    #               outputfile=r'D:\data\Geology\GLHYMPS 2.0\tif\logK_Ice_x1.tif',
    #               templatefile=r'C:\Users\admini\Desktop\0.25.tif',
    #               field='logK_Ice_x')

    # resample_images(path_refer=r'C:\Users\admini\Desktop\0.25.tif',
    #                 path_resample=r'D:\data\Geology\GLHYMPS 2.0\tif\logK_Ice_x.tif',
    #                 out_path_resample=r'D:\data\Geology\GLHYMPS 2.0\tif\logK_Ice_x2.tif')

    # resample_images(path_refer=r'C:\Users\admini\Desktop\0.25.tif',
    #                 path_resample=r'D:\data\Geology\GLHYMPS 2.0\tif\Porosity_x.tif',
    #                 out_path_resample=r'D:\data\Geology\GLHYMPS 2.0\tif\Porosity_x2.tif')

    # resample_images(path_refer=r'C:\Users\admini\Desktop\0.25.tif',
    #                 path_resample=r'D:\data\Soil\Depth to rock\BDTICM_M_250m_ll.tif',
    #                 out_path_resample=r'D:\data\Soil\Depth to rock\0.25\BDTICM_M_250m_ll.tif')

    # resample_images(path_refer=r'C:\Users\admini\Desktop\0.25.tif',
    #                 path_resample=r'D:\data\Soil\Soil and sedimentary deposit thickness\average_soil_and_sedimentary-deposit_thickness.tif',
    #                 out_path_resample=r'D:\data\Soil\Soil and sedimentary deposit thickness\0.25\deposit_thickness.tif')

    # resample_images(path_refer=r'C:\Users\admini\Desktop\0.25.tif',
    #                 path_resample=r'D:\data\Soil\Soil hydraulic parameters\k_s\log_k_s.tif',
    #                 out_path_resample=r'D:\data\Soil\Soil hydraulic parameters\k_s\0.25\log_k_s.tif')
    # resample_images(path_refer=r'C:\Users\admini\Desktop\0.25.tif',
    #                 path_resample=r'D:\data\Soil\Soil hydraulic parameters\k_s\log_k_s_s.tif',
    #                 out_path_resample=r'D:\data\Soil\Soil hydraulic parameters\k_s\0.25\log_k_s_s.tif')
    # resample_images(path_refer=r'C:\Users\admini\Desktop\0.25.tif',
    #                 path_resample=r'D:\data\Geology\GLHYMPS 2.0\tif\origin\logK_Ice_x.tif',
    #                 out_path_resample=r'D:\data\Geology\GLHYMPS 2.0\tif\logK_Ice_x.tif')

    # soil physical
    # mosaic_tif(r'D:\data\soil\Soil physical parameters\clay content',
    #            r'D:\data\soil\Soil physical parameters\clay content\clay content.tif')
    # resample_tif(r'D:\data\soil\Soil physical parameters\clay content\clay content.tif',
    #              r'D:\data\soil\Soil physical parameters\clay content.tif')
    # mosaic_tif(r'D:\data\soil\Soil physical parameters\sand content',
    #            r'D:\data\soil\Soil physical parameters\sand content\sand content.tif')
    # resample_tif(r'D:\data\soil\Soil physical parameters\sand content\sand content.tif',
    #              r'D:\data\soil\Soil physical parameters\sand content.tif')
    # mosaic_tif(r'D:\data\soil\Soil physical parameters\silt content',
    #            r'D:\data\soil\Soil physical parameters\silt content\silt content.tif')
    # resample_tif(r'D:\data\soil\Soil physical parameters\silt content\silt content.tif',
    #              r'D:\data\soil\Soil physical parameters\silt content.tif')

    # human
    # resample_tif(r'D:\data\surface\human\Road Density\grip4_total_dens_m_km2.tif',
    #              r'D:\data\surface\human\Road Density\grip4_total_dens_m_km2_25.tif')
    # resample_tif(r'D:\data\surface\human\GDP\GDP_PPP_001.tif',
    #              r'D:\data\surface\human\GDP\GDP_PPP_025.tif')

    # dem
    resample_tif(r'D:\data\tibet\dem\tibet_dem.tif',
                 r'D:\data\tibet\dem\tibet_dem_25.tif')

    # dem
    # resample_tif(r'D:\data\tibet\dem\tibet_dem.tif',
    #              r'D:\data\tibet\dem\tibet_dem_25.tif')
