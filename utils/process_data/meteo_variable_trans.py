import numpy as np


class MeteoVariableTrans:
    @staticmethod
    # uv风速转为总风速m/s
    def uv_to_ws(u: np.ndarray, v: np.ndarray):
        return np.sqrt(np.square(u) + np.square(v))

    @staticmethod
    # 日降水量mm/day 转为 降水率kg/(m2*s)
    # 1 kg/(m2*s) = 86400 mm/day
    def tp_to_pr(tp: np.ndarray):
        return tp / 86400

    @staticmethod
    # 降水率kg/(m2*s) 转为 日降水量mm/day
    def pr_to_tp(pr: np.ndarray):
        return pr * 86400

    @staticmethod
    # 露点温度K 转为 空气相对湿度%
    # hurs 近地表空气相对湿度 = 蒸汽压 / 饱和蒸汽压
    def dem_to_hurs(dem: np.ndarray, tem: np.ndarray):
        a, b = 17.67, 29.65  # Tetens公式
        hurs = np.exp(a * (dem - 273.15) / (dem - b) - a * (tem - 273.15) / (tem - b))
        return hurs * 100

    @staticmethod
    # 露点温度K 转为 空气比湿%
    # huss 近地表空气比湿 = 水汽的质量 / 空气总质量
    def dem_to_huss(dem: np.ndarray, sp: np.ndarray):
        a, b, es0 = 17.269, 35.86, 6.1078
        sp = sp / 100
        e = es0 * np.exp(a * (dem - 273.15) / (dem - b))
        q = 0.622 * e / (sp - 0.378 * e)
        return q

    @staticmethod
    # 比湿kg/kg 转为 相对湿度%
    # tem为气温K，sp为近地表气压Pa
    def huss_to_hurs(huss: np.ndarray, tem: np.ndarray, sp: np.ndarray):
        hurs = 0.263 * sp * huss * np.exp((17.67 * (tem - 273.15)) / (tem - 29.65)) ** (-1)
        return hurs

    @staticmethod
    # 相对湿度% 转为 比湿kg/kg
    # tem为气温K，sp为近地表气压Pa
    def hurs_to_huss(hurs: np.ndarray, tem: np.ndarray, sp: np.ndarray):
        huss = hurs / 0.263 * sp * np.exp((17.67 * (tem - 273.15)) / (tem - 29.65)) ** (-1)
        return huss

    @staticmethod
    # 近地表气压pa 转为 海平面气压pa，tm为温度气柱压力
    def sp_to_psl(sp: np.ndarray, height: np.ndarray, tem: np.ndarray):
        height = height.reshape(1, height.shape[0], height.shape[1]).repeat(sp.shape[0], axis=0)
        tm = (tem + 0.0025 * height + tem) / 2 - 273.15
        return sp * np.power(10, height / 18400 / (1 + tm / 273))

    @staticmethod
    # 海平面气压pa 转为 近地表气压pa，tm为温度气柱压力，height为海拔
    def psl_to_sp(psl: np.ndarray, height: np.ndarray, tem: np.ndarray):
        height = height.reshape(1, height.shape[0], height.shape[1]).repeat(psl.shape[0], axis=0)
        tm = (tem + 0.0025 * height + tem) / 2 - 273.15
        return psl / np.power(10, height / 18400 / (1 + tm / 273))

    @staticmethod
    # 辐射能量J/(m2*day) 转为 辐射功率W/(m2*s)
    def rad_energy_to_power(energy: np.ndarray):
        return energy / 86400


if __name__ == '__main__':
    # a = era5_to_cmip6.ps_to_psl(np.array([54545]), np.array([3545]), np.array([284]))
    # a = era5_to_cmip6.rad_energy_to_power(np.array([9232142]))
    a = MeteoVariableTrans.dem_to_huss(np.array([247]), np.array([88476]))
    b = MeteoVariableTrans.dem_to_hurs(np.array([247]), np.array([260]))
    c = MeteoVariableTrans.huss_to_hurs(a, np.array([260]), np.array([88476]))
    d = MeteoVariableTrans.huss_to_hurs(np.array([0.000838]), np.array([260]), np.array([88476]))
    print(a)
    print(b)
    print(c)
    print(d)
