import os


class PathConfig:
    caravan_path = r'D:\data\Caravan'
    caravan_timeseries_path = os.path.join(caravan_path, r'timeseries\csv')
    caravan_attributes_path = os.path.join(caravan_path, r'attributes')
    caravan_shapefiles_path = os.path.join(caravan_path, r'shapefiles')

    experiment_path = r'D:\experiment\data'
    # experiment_path = r'/home/cas-519/storage-2t/wzl/experiment/data'
    origin_path = os.path.join(experiment_path, 'origin_data')
    train_path = os.path.join(experiment_path, 'train_data')

    model_path = r'D:\experiment\model'
    # model_path = r'/home/cas-519/storage-2t/wzl/experiment/model'
    model_conf_path = os.path.join(model_path, 'config')

    camels_list = ['camels', 'camelsaus', 'camelsbr',
                   'camelscl', 'camelsgb', 'hysets', 'lamah']

    shp_combined_path = os.path.join(caravan_shapefiles_path, r'combined.shp')
