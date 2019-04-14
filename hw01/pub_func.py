import numpy as np
import pandas as pd

def read_raw_data(f_path, f_type='train', hour_range=9, hours_each_day=24):
    hour = hours_each_day
    hour_range = hour_range
    raw_features = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx',
                'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC',
                'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
    col_names_in_test = ['date', 'item'] + list(map(str, range(9)))
    if f_type == 'train':
        print('This is training set...')
        # read raw data and label it
        # 1...24 means hour of that day
        colnames = ['date', 'site', 'item'] + [str(i) for i in range(24)]
        # skip all Chinese characters to avoid encoding probole
        raw_data = pd.read_csv('./given/train.csv', names=colnames, skiprows=1, usecols=[0] + list(range(2,27)))
        # raw_data = pd.read_csv(f_path, delimiter=',', encoding='big5') ## train.csv
    elif f_type == 'test':
        print('This is test set...')
        raw_data = pd.read_csv(f_path, header=None, names=col_names_in_test) ## test.csv

    hour2raw_feature = pd.DataFrame(columns=raw_features + ['day_index', 'hour_index', 'month_index'])
    
    # convert raw data to raw features of each hour (each hour as raws and each raw feature as columns)
    print('Start to convert raw data to each hour x raw features...')
    days = list(np.unique(raw_data['date']))

    for i, day in enumerate(days):
        _current_day = raw_data.loc[raw_data['date']==day, :]
        for h in range(hour): # for each hour in current day
            hour2raw_feature.loc[i*hour + h, raw_features] = list(_current_day.loc[:, str(h)])
            hour2raw_feature.loc[i*hour + h, 'hour_index'] = h
            if f_type == 'train':
                hour2raw_feature.loc[i*hour + h, 'day_index'] = i
                hour2raw_feature.loc[i*hour + h, 'month_index'] = day.split('/')[1]
            elif f_type == 'test':
                # the records of days in each month are not continuous, so set each day as different month
                day_id = int(day.split('_')[1])
                hour2raw_feature.loc[i*hour + h, 'day_index'] = day_id
                hour2raw_feature.loc[i*hour + h, 'month_index'] = str(day_id + 1) # start from 1, same as training set
            if hour2raw_feature.loc[i*hour + h, 'RAINFALL'] == 'NR':
                hour2raw_feature.loc[i*hour + h, 'RAINFALL'] = 0
    hour2raw_feature[raw_features] = hour2raw_feature[raw_features].astype(float)
    
    # ['AMB_TEMP_0', 'CH4_0', ..., 'WS_HR_0', ...,'AMB_TEMP_8', 'CH4_8', ..., 'WS_HR_8']
    real_features = [i + '_' + str(j) for j in range(hour_range) for i in raw_features]
    real_features2y = pd.DataFrame(columns=real_features + ['y'])
    
    month = len(np.unique(hour2raw_feature['month_index']))
    print('Start to create real feature...')
    for m in range(1, month+1):  # 12 month in training set, 240 month (same as days) in test set
        # print(m)
        _current_month = hour2raw_feature.loc[hour2raw_feature['month_index']==str(m), :]
        number_of_hours = _current_month.shape[0]
        steps = number_of_hours - hour_range
        for s in range(steps): # not execute in test set
            _current_features = _current_month.iloc[range(s,s+hour_range), range(len(raw_features))]
            _current_y = _current_month.iloc[s+hour_range, raw_features.index('PM2.5')]
            # print(_current_y)
            # flatten by row
            real_features2y.loc[(m-1)*steps + s, :] = np.c_[_current_features.values.flatten().reshape(1, -1), 
                                                            _current_y.reshape(1, -1)]
        if f_type == 'test':
            _current_features = _current_month.iloc[range(hour_range), range(len(raw_features))]
            _current_y = None
            # print(_current_y)
            # flatten by row
            real_features2y.loc[m-1, :] = np.c_[_current_features.values.flatten().reshape(1, -1), 
                                                _current_y]
    print('The shape of data table: ', real_features2y.shape)
    return real_features2y