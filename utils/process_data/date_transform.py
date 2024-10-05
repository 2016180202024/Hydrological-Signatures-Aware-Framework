import numpy as np
import datetime as dt


def date_to_year(date_array):
    year_array = []
    for date in date_array:
        year = date[0:4]
        year_array.append(year)
    return np.array(year_array)


def date_to_doy(date: dt.date):
    d1 = date.toordinal()  # 获取该日期在公历中的序数
    d0 = dt.date(date.year, 1, 1).toordinal()  # 这一年1月1日的序数
    return d1 - d0 + 1


def get_days_in_year(year: int):
    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year+1, 1, 1)
    days = (end_date - start_date).days
    return days
