import pandas as pd

def getMonth(_str):
    return int(_str.split("-")[1])

def getYear(_str):
    return int(_str.split("-")[0])

def getDay(_str):
    return int(_str.split("-")[2])


if __name__=='__main__':
    data_file = '../utils/data/temp_electric_holiday_increase.csv'
    data = pd.read_csv(data_file)
    data['month']  = data['date'].apply(getMonth)
    data['day'] = data['date'].apply(getDay)
    data['year'] = data['date'].apply(getYear)
    data = data[['day', 'month', 'year', 'holiday','temp', 'increase', 'power']]
    data.to_csv('../utils/data/power/month_day_year_temp_increase_power.csv')
