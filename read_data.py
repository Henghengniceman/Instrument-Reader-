# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:36:04 2021

@author: st5536

Python module to read out various instruments.
"""

import typing
import pandas as pd
import os


def read_aps_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    meta_data = pd.read_table(file_name, sep=",", on_bad_lines='skip', encoding='latin1')
    df = pd.read_csv(file_name, sep=',', header=len(meta_data)+1, encoding='latin1')

    df.columns = ['Sample #', 'Date', 'Start Time', 'Aerodynamic Diameter',
                  float(meta_data.iloc[4, 1]),
                  0.542, 0.583, 0.626, 0.673, 0.723, 0.777, 0.835, 0.898,
                  0.965, 1.037, 1.114, 1.197, 1.286, 1.382, 1.486, 1.596,
                  1.715, 1.843, 1.981, 2.129, 2.288, 2.458, 2.642, 2.839,
                  3.051, 3.278, 3.523, 3.786, 4.068, 4.371, 4.698, 5.048,
                  5.425, 5.829, 6.264, 6.732, 7.234, 7.774, 8.354, 8.977,
                  9.647, 10.37, 11.14, 11.97, 12.86, 13.82, 14.86, 15.96,
                  17.15, 18.43, 19.81, 'Event 1', 'Event 3', 'Event 4', 'Dead Time',
                  'Inlet Pressure', 'Total Flow', 'Sheath Flow', 'Analog Input Voltage 0',
                  'Analog Input Voltage 1', 'Digital Input Level 0',
                  'Digital Input Level 1', 'Digital Input Level 2', 'Laser Power',
                  'Laser Current', 'Sheath Pump Voltage', 'Total Pump Voltage',
                  'Box Temperature', 'Avalanch Photo Diode Temperature',
                  'Avalanch Photo Diode Voltage', 'Status Flags', 'Median(µm)',
                  'Mean(µm)', 'Geo. Mean(µm)', 'Mode(µm)', 'Geo. Std. Dev.',
                  'Total Conc.']

    df['DateTime'] = pd.to_datetime(df['Date']
                                    + ' '
                                    + df['Start Time'],
                                    format='%m/%d/%y %H:%M:%S',
                                    errors='coerce')
    df = df[~pd.isnull(df['DateTime'])]
    df.index = df['DateTime']
    df.drop('DateTime', axis=1, inplace=True)
    df['Total Conc.'] = ((df['Total Conc.']
                          .str.extract(r'[+-]?(\d+([.]\d*)?([eE][+-]?\d+)?'
                                       r'|[.]\d+([eE][+-]?\d+)?)\((.*?)\)')[0])
                         .astype(float))
    df = df.sort_index()
    df.attrs = {f'{meta_data.columns[0]}': meta_data.columns[1],
                f'{meta_data.iloc[0, 0]}': float(meta_data.iloc[0, 1]),
                f'{meta_data.iloc[1, 0]}': float(meta_data.iloc[1, 1]),
                f'{meta_data.iloc[2, 0]}': meta_data.iloc[2, 1],
                f'{meta_data.iloc[3, 0]}': meta_data.iloc[3, 1],
                f'{meta_data.iloc[4, 0]}': float(meta_data.iloc[4, 1]),
                f'{meta_data.iloc[5, 0]}': float(meta_data.iloc[5, 1])}
    return df


def read_cpc3010_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    df = pd.read_csv(file_name, names=['Time', 'Concentration / cm-3'],
                     delimiter='\t', parse_dates=['Time'])
    df.rename(columns={'Time': 'time',
                       'Concentration / cm-3': 'concentration / cm-3'},
              inplace=True)
    df.set_index('time', inplace=True)
    return df


def read_cpc3772_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(pd.read_table(file_name, encoding='latin1')
                      .iloc[:, 0].str.split(',', expand=True).to_numpy())
    df = df[~df.iloc[:, 0].str.contains('Comment')]
    metadata = df.iloc[:13, :]
    start_date = metadata.iloc[2, 1:-1:2].to_list()
    start_time = metadata.iloc[3, 1:-1:2].to_list()
    start_datetime = [pd.to_datetime(date + time, format='%m/%d/%y%H:%M:%S')
                      for date, time in zip(start_date, start_time)]

    data_df = df.iloc[13:, :]
    column_lst = data_df.iloc[0, :].to_list()
    data_df.rename(columns=dict(zip(data_df.columns, column_lst)), inplace=True)
    data_df.drop(data_df.index[0], inplace=True)

    time_elapsed = data_df['Elapsed (s)'].astype(float)

    data_lst = []
    for j in range(len(data_df.columns)//3):
        _data = data_df.iloc[:, j*2+1:j*2+3]
        _data['DateTime'] = time_elapsed.map(lambda x: start_datetime[j] + pd.Timedelta(seconds=x))
        data_lst.append(_data)

    full_df = pd.concat(data_lst)

    full_df.index = pd.to_datetime(full_df['DateTime'])
    full_df.drop(['DateTime'], axis=1, inplace=True)
    full_df.rename(columns={'Concentration (#/cm³)': 'concentration / cm-3',
                            'Count (#)': 'count / -'}, inplace=True)
    full_df['concentration / cm-3'].replace('', np.nan, inplace=True)
    full_df['count / -'].replace('', np.nan, inplace=True)
    df = full_df.astype(float)
    return df


def read_cpc3776_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import pathlib

    if pathlib.Path(file_name).suffix == '.csv':
        df = pd.DataFrame(pd.read_table(file_name, encoding='latin1')
                          .iloc[:, 0].str.split(',', expand=True).to_numpy())
        dates = pd.to_datetime(df.loc[df.iloc[:, 0] == 'Start Date', 1].to_numpy()[0]
                               + df.loc[df.iloc[:, 0] == 'Start Time', 1].to_numpy()[0],
                               format='%m/%d/%y%H:%M:%S')
        row_idx = df.index[df.iloc[:, 0] == '1.0'].to_numpy()[0]
        cpc_data = df.iloc[row_idx:-1, :5]
        for col in cpc_data.columns:
            cpc_data.loc[:, col] = cpc_data.loc[:, col].astype(float)
        cpc_data.columns = ['elapsed / s', 'concentration / cm-3', 'count / -',
                            'Analog1', 'Analog2']
        cpc_data.index = pd.to_datetime(cpc_data['clapsed / s'], unit='s',
                                        origin=dates)
        cpc_data.index.name = 'DateTime'
        cpc_data.drop('elapsed / s', axis=1, inplace=True)
        return cpc_data

    elif pathlib.Path(file_name).suffix == '.txt':
        dates = pd.read_csv(file_name, sep=',', skiprows=3,
                            on_bad_lines='skip',
                            encoding='latin1',
                            engine='python')
        date_list = dates.iloc[:, 1::2].to_numpy()
        lst = date_list[0][:-1:]

        df = pd.read_csv(file_name, sep=",", header=14, encoding='latin1')
        df = df[~df['Time'].str.contains("Comment")]
        for j, element in enumerate(lst):
            print('j, element \n', j, element)
            if j == 0:
                df['Time'] = pd.to_datetime(element + ' ' + df['Time'], format='%m/%d/%y %H:%M:%S')
            else:
                df[f'Time.{j}'] = pd.to_datetime(element + ' ' + df[f'Time.{j}'],
                                                 format='%m/%d/%y %H:%M:%S')

        df_conc = df[df.columns[pd.Series(df.columns).str.startswith('Concentration')]]
        df_time = df[df.columns[pd.Series(df.columns).str.startswith('Time')]]
        df_conc_list = pd.concat([df_conc[col] for col in df_conc.columns])
        df_time_list = pd.concat([df_time[col] for col in df_time.columns])
        df = pd.DataFrame({'concentration / cm-3': df_conc_list,
                           'time': df_time_list})
        df = df[df['concentration / cm-3'].str.strip().astype(bool)]
        df['concentration / cm-3'] = df['concentration / cm-3'].astype(float)
        df = df.loc[df['time'].notnull()]
        df.index = df['time']
        df = df.sort_index()
        df.drop(['time'], axis=1, inplace=True)
        return df


def read_cpc3752_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    meta_data = pd.read_table(file_name, sep=',', nrows=14, header=None)

    df = pd.read_csv(file_name, header=15, sep=',', parse_dates=['Date-Time'], low_memory=False)
    df.index = df['Date-Time']
    df.index.name = 'time'
    df.rename(columns={'Concentration (#/cm3)': 'concentration / cm-3'},
              inplace=True)

    try:
        df = df[~df['Error'].str.contains('Error|Warn', na=False, regex=True)]
    except AttributeError as e:
        print(f'{e} --> No Error or Warn in Error columns.')
    df.drop(['Date-Time', 'Elapsed Time(s)', 'Error', 'Unnamed: 5'], axis=1, inplace=True)
    df = df.sort_index()

    df.attrs = meta_data.set_index(0)[1].to_dict()

    return df


def read_fidas_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    # PDAnalyze v2.037, load file, tick size distribution, number of intervals / decade should be
    # 128 and lower and upper limit should be 0.1 and 200 um, respectively.
    import numpy as np
    import pandas as pd

    with open(file_name) as f:
        lines = f.readlines()
        ncols = max([len(lines[i].split('\t')) for i in range(len(lines))])

    df = pd.read_table(file_name, sep=r'\t', header=None, names=np.arange(0, ncols-1),
                       decimal=',', skipfooter=1, engine='python', encoding='ANSI')
    index = min([i for i, _ in enumerate(df.iloc[1, :]) if _ == ' '])
    lower_bin_bounds = df.iloc[1, :index].astype(float)
    mean_bin_bounds = df.iloc[2, :index].astype(float)
    upper_bin_bounds = df.iloc[3, :index].astype(float)

    bin_boundaries = sorted(list(set(lower_bin_bounds) | set(upper_bin_bounds)))
    dlogdp = [np.log10(bin_boundaries[i+1]) - np.log10(bin_boundaries[i])
              for i in range(len(bin_boundaries) - 1)]
    df_dN = df.iloc[4:, :index+6]
    df_dN.index = pd.to_datetime(df_dN.iloc[:, 0] + df_dN.iloc[:, 1],
                                 format='%d.%m.%Y%H:%M:%S')
    df_dN.index.name = 'time'
    df_dN.drop([0, 1, 4, 5], axis=1, inplace=True)

    col_lst = ['time since start / s', 'duration of interval / s'] + mean_bin_bounds.to_list()
    for i, col in enumerate(df_dN.columns):
        df_dN.loc[:, col] = df_dN.loc[:, col].astype(float)

    df_dN.columns = col_lst
    df_dN.attrs = {'lower_bin_bounds': lower_bin_bounds.to_numpy(),
                   'mean_bin_bounds': mean_bin_bounds.to_numpy(),
                   'upper_bin_bounds': upper_bin_bounds.to_numpy(),
                   'dlogdp': np.array(dlogdp)}

    df_dN = df_dN[(df_dN.index > '2000-01-01 00:00:00')]
    df_dN = df_dN.sort_index()

    return df_dN


def read_fidas_PM_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_table(file_name, header=4, skipfooter=6, decimal=',',
                       engine='python', encoding='ANSI')
    df.index = pd.to_datetime(df['date beginning'] + df['time beginning'],
                              format='%d.%m.%Y%H:%M:%S')
    df.index.name = 'time'
    df.drop(['date beginning', 'time beginning', 'date end', 'time end', 'relative time [s]',
             'Unnamed: 9'],
            axis=1, inplace=True)
    df.columns = ['PM1 / µg m-3', 'PM2_5 / µg m-3', 'PM4 / µg m-3', 'PM10 / µg m-3']
    return df


def read_ufcpc_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(file_name, sep='\t', decimal=',', engine='python',
                     usecols=['date', 'time', 'relative time [s]', 'Cn [P/cm^3]'])
    df.index = pd.to_datetime(df['date'] + df['time'], format='%d.%m.%Y%H:%M:%S')
    df.drop(['date', 'time'], axis=1, inplace=True)
    df.columns = ['relative time / s', 'concentration / cm-3']
    df.index.name = 'time'
    return df


def read_ws700_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(file_name, sep='\t', encoding='ANSI')
    df = df.loc[df['error'] == 0]
    df.index = pd.to_datetime(df['datetime'])
    df.drop(['datetime'], axis=1, inplace=True)
    df.sort_index(inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(float)
    return df


def read_cpcai_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(file_name, sep='\t', encoding='ANSI')

    df = df.loc[df['error'] == 0]
    df.index = pd.to_datetime(df['datetime'])
    df.drop(['datetime', '2021-05-04 13:55:05', '10^(x-3)*1.2', '200/1*x',
             '50/10*x', '2000/10*x', '100/10*x', '100/10*x.1', '100/10*x.2',
             '100000/10*x'], axis=1, inplace=True)
    df.sort_index(inplace=True)
    return df


def read_usmps_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    with open(file_name) as f:
        lines = f.readlines()
        ncols = max([len(lines[i].split('\t')) for i in range(len(lines))])

    df = pd.read_table(file_name, sep=r'\t', header=None, names=np.arange(0, ncols-1),
                       decimal=',', skipfooter=1, engine='python', encoding='ANSI')
    index = min([i for i, _ in enumerate(df.iloc[1, :]) if _ == ' '])
    lower_bin_bounds = df.iloc[1, :index].astype(float)
    mean_bin_bounds = df.iloc[2, :index].astype(float)
    upper_bin_bounds = df.iloc[3, :index].astype(float)

    bin_boundaries = sorted(list(set(lower_bin_bounds) | set(upper_bin_bounds)))
    dlogdp = [np.log10(bin_boundaries[i+1]) - np.log10(bin_boundaries[i])
              for i in range(len(bin_boundaries) - 1)]
    df_dN = df.iloc[4:, :index+6]
    df_dN.index = pd.to_datetime(df_dN.iloc[:, 0] + df_dN.iloc[:, 1],
                                 format='%d.%m.%Y%H:%M:%S')
    df_dN.index.name = 'DateTime'
    df_dN.drop([0, 1, 4, 5], axis=1, inplace=True)

    col_lst = ['time since start / s', 'duration of interval / s'] + mean_bin_bounds.to_list()
    for i, col in enumerate(df_dN.columns):
        df_dN.loc[:, col] = df_dN.loc[:, col].astype(float)

    df_dN.columns = col_lst
    df_dN.attrs = {'lower_bin_bounds': lower_bin_bounds.to_numpy(),
                   'mean_bin_bounds': mean_bin_bounds.to_numpy(),
                   'upper_bin_bounds': upper_bin_bounds.to_numpy(),
                   'dlogdp': np.array(dlogdp)}

    df_dN = df_dN[(df_dN.index > '2000-01-01 00:00:00')]
    df_dN = df_dN.sort_index()

    return df_dN


def read_smps_1_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    # TODO: Add metadata and description.
    idx_lst, header_lst = [0], [0]
    df = pd.read_table(file_name, encoding='ANSI')
    for idx in df.index:
        if df.iloc[idx, 0].startswith('Sample File'):
            idx_lst.append(idx)
        if df.iloc[idx, 0].startswith('Sample #'):
            header_lst.append(idx)

    df_lst = []
    for i in range(len(idx_lst)):
        if i == len(idx_lst) - 1:
            element = df.iloc[idx_lst[i]:, :]
        else:
            element = df.iloc[idx_lst[i]:idx_lst[i+1], :]
        df_lst.append(element)

    data_lst = []
    for i, df in enumerate(df_lst[:]):
        df_data = df.loc[header_lst[i+1]:, df.columns[0]].str.split(',', expand=True)
        new_columns = df_data.iloc[0, :].str.strip().apply(pd.to_numeric, errors='ignore')
        df_data.rename(columns=dict(zip(df_data.columns, new_columns)), inplace=True)
        df_data.drop(df_data.index[0], inplace=True)
        df_data.index = pd.to_datetime(df_data['Date'] + df_data['Start Time'],
                                       format='%m/%d/%y%H:%M:%S')
        data_lst.append(df_data)

    combined_data = pd.concat(data_lst)

    return combined_data


def read_smps_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    # TODO: add meta data dictionary
    meta_data = pd.read_table(file_name, skiprows=lambda x: x > 22,names=np.arange(0, 12))

    df = pd.read_table(file_name, skiprows=23)

    df.insert(0, 'Start DateTime', (pd.to_datetime(df['Start Year'], format='%Y')
                                    + df['Start Date'].map(lambda x: pd.Timedelta(days=x))
                                    ).round('1s'))
    df.insert(1, 'End DateTime', (pd.to_datetime(df['End Year'], format='%Y')
                                  + df['End Date'].map(lambda x: pd.Timedelta(days=x))
                                  ).round('1s'))
    df.index = df['End DateTime'] - (df['End DateTime'] - df['Start DateTime']) / 2
    # Drop not needed date and year information
    df.drop(['Start Date', 'Start Year', 'End Date', 'End Year'], axis=1, inplace=True)
    # Drop secondary columns that contain the cumulative distribution
    df.drop(list(df.filter(regex=r'\.[0-9]{1,}\.')), axis=1, inplace=True)

    # Create a column list to transform some values to floats and rename the columns afterwards
    col_lst = []
    for i, col in enumerate(df.columns):
        try:
            val = float(col)
        except ValueError:
            val = col
        col_lst.append(val)

    df.rename(columns=dict(zip(df.columns, col_lst)), inplace=True)

    df_sd = df.iloc[:, [isinstance(_, float) for _ in df.columns]]
    df_sd.columns = df_sd.columns.astype(float)

    df_dct = {'full': df,
              'sd': df_sd}

    return df_dct


def read_aps_data_sammal(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    diag_file_name = file_name[:3] + '_diag' + file_name[3:]

    data = pd.read_csv(file_name, header=1)
    data.index = pd.to_datetime(data.index.get_level_values(level=0))
    data.drop('0', axis=1, inplace=True)
    data.columns = [float(x) for x in data.columns]
    idx = data.index[:]
    diag_data = pd.read_csv(diag_file_name, header=None)
    diag_data.index = pd.to_datetime(diag_data[0], format='%Y%m%d%H%M%S')
    diag_data.drop(0, axis=1, inplace=True)
    diag_data.columns = ['0', '1', '2', '3', 'ambient pressure / hPa', '5', '6',
                         'sample flow / lpm', 'sheath flow / lpm', 'total flow / lpm', '10', '11',
                         '12']
    df = data.combine_first(diag_data['sample flow / lpm'].to_frame()).ffill().bfill()
    df = df.loc[idx, :]
    df = df.iloc[:, :-1].div(df['sample flow / lpm'] * 1e3, axis=0)
    return df


def read_opcN3_firmware(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_table(file_name).iloc[:, 0].str.split(',', expand=True)
    idx = df[df.loc[:, 0] == 'Data:'].index.to_numpy()[0]
    metadata = df.iloc[:idx, :]
    df.attrs = {'Device SerNo': metadata.iloc[0, 1].rstrip(),
                'InfoString': metadata.iloc[1, 1],
                'Laser digital pot setting': metadata.iloc[2, 1],
                'Fan digital pot setting': metadata.iloc[3, 1],
                'ToF to SFR factor': metadata.iloc[4, 1],
                'Bins': metadata.iloc[5, 1:26].to_numpy(),
                'Bin low boundary (ADC o/p)': metadata.iloc[6, 1:26].to_numpy(),
                'Bin low boundary (particle diameter [um])':
                    pd.to_numeric(metadata.iloc[7, 1:26].to_numpy()),
                'Bin mean (particle diameter [um])': metadata.iloc[8, 1:26].to_numpy(),
                'Vol of a particle in bin (um3)': metadata.iloc[9, 1:26].to_numpy(),
                'Weighting for bin': metadata.iloc[10, 1:26].to_numpy()
                }
    columns = df.iloc[idx+1, :]
    data = df.iloc[idx+2:, :]
    data.columns = columns
    data.index = pd.to_datetime(data['OADateTime'].astype(float), unit='d',
                                origin=pd.Timestamp('1899-12-30'))
    data.index.name = 'DateTime'
    data.drop('OADateTime', axis=1, inplace=True)
    for col in data.columns:
        data.loc[:, col] = data.loc[:, col].astype(float)

    return data


def read_opcN3_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_name, parse_dates=['DateTime'])

    df.columns = ['DateTime', 0.35, 0.46, 0.66, 1.00, 1.30, 1.70, 2.30, 3.00, 4.00, 5.20, 6.50,
                  8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 25.00, 28.00, 31.00,
                  34.00, 37.00, 'Bin1 MToF / us', 'Bin3 MToF / us', 'Bin5 MToF / us',
                  'Bin7 MToF / us', 'Sampling period / s', 'FlowRate / ml/s',
                  'OPC-T / °C', 'OPC-RH / %RH', 'PM1 / ug/m3', 'PM2.5 / ug/m3',
                  'PM10 / ug/m3', 'Reject count glitch', 'Reject count LongToF',
                  'Reject count Ratio', 'Reject count OutOfRange', 'Fan Rev Count',
                  'Laser status', 'Checksum', 'Checksum_calc']
    df.index = df['DateTime']
    df.insert(loc=df.columns.get_loc('Bin1 MToF / us'), column=40.00, value=np.nan)

    return df


def read_UHSAS_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    def am_pm_addition(series, start_time):
        series = series.map(lambda x: pd.to_datetime(x.rstrip(), format='%H:%M:%S.%f'))
        series = series.map(lambda dt: dt.replace(year=start_time.year, month=start_time.month,
                                                  day=start_time.day))
        time_list = []
        for idx in series.index:
            if start_time.hour < 12:
                try:
                    if idx < series[series.dt.hour == 12].index[0]:
                        time_list.append('AM')
                    else:
                        time_list.append('PM')
                except IndexError:
                    time_list.append('AM')
            else:
                try:
                    if idx < series[series.dt.hour == 12].index[0]:
                        time_list.append('PM')
                    else:
                        time_list.append('AM')
                except IndexError:
                    time_list.append('PM')
        series = series.dt.strftime('%Y-%m-%d %H:%M:%S.%f') + time_list
        return series

    start_time = pd.to_datetime(file_name.stem, format='%Y%m%d%H%M%S')
    df = pd.read_table(file_name, low_memory=False)
    lower_bin_boundaries = df.columns[15:].to_numpy(dtype=float)
    bin_boundaries = np.append(lower_bin_boundaries, 1000)
    df.drop(0, inplace=True)
    df['Time'] = am_pm_addition(df['Time'], start_time)
    df.index = pd.to_datetime(df['Time'],
                              format='%Y-%m-%d %I:%M:%S.%f%p')
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    df.columns = [pd.to_numeric(x, errors='ignore') for x in df.columns]
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].astype(float)

    df['Sample_volume'] = (df['Accum.'] / 60 * df['Sample'])
    dlogdp = [np.log10(bin_boundaries[i+1]) - np.log10(bin_boundaries[i])
              for i in range(len(bin_boundaries) - 1)]
    df.attrs = {'dlogdp': dlogdp}

    df = df.sort_index()
    return df


def read_opcN3_Pi_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_name)
    df.index = pd.to_datetime(df['DateTime'])

    df.columns = ['DateTime', 0.35, 0.46, 0.66, 1.00, 1.30, 1.70, 2.30, 3.00, 4.00, 5.20, 6.50,
                  8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 25.00, 28.00, 31.00,
                  34.00, 37.00, 'Bin1 MToF / us', 'Bin3 MToF / us', 'Bin5 MToF / us',
                  'Bin7 MToF / us', 'Sampling period / s', 'FlowRate / ml/s',
                  'OPC-T / °C', 'OPC-RH / %RH', 'PM1 / ug/m3', 'PM2.5 / ug/m3',
                  'PM10 / ug/m3', 'Reject count glitch', 'Reject count LongToF',
                  'Reject count Ratio', 'Reject count OutOfRange', 'Fan Rev Count',
                  'Laser status', 'Checksum', 'Checksum_calc']
    df.insert(loc=df.columns.get_loc('Bin1 MToF / us'), column=40.00, value=np.nan)

    return df


def read_BME280_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(file_name, names=['DateTime', 'T / °C', 'p / hPa', 'RH / %'],
                     parse_dates=['DateTime'])
    df.set_index(['DateTime'], inplace=True)

    return df


def read_SHT40_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(file_name, names=['DateTime', 'T / °C', 'RH / %'],
                     parse_dates=['DateTime'])
    df.set_index(['DateTime'], inplace=True)

    return df


def read_SFM4100_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(file_name, header=0,
                     parse_dates=['UTC DateTime'])
    df.set_index(['UTC DateTime'], inplace=True)

    return df


def read_UCASS_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import re

    metadata = pd.read_table(file_name, skiprows=lambda x: x not in range(4))
    df = pd.read_csv(file_name, header=4, parse_dates=['UTC DateTime'])
    df.set_index(['UTC DateTime'], inplace=True)
    df.attrs = {
        f'{metadata.columns[0].split(",")[0]}': f'{metadata.columns[0].split(",")[1]}',
        f'{metadata.iloc[0, 0].split(",")[0]}': f'{metadata.iloc[0, 0].split(",")[1]}',
        'Bins': metadata.iloc[1, 0].split(",")[:17],
        'ADC': [int(_) for _ in re.findall(r'".+?"|[\w-]+', metadata.iloc[2, 0])[:16]],
        f'{metadata.iloc[1, 0].split(",")[17]}':
            re.findall(r'".+?"|[\w-]+', metadata.iloc[2, 0])[16],
        f'{metadata.iloc[1, 0].split(",")[18]}':
            re.findall(r'".+?"|[\w-]+', metadata.iloc[2, 0])[17],
        }

    return df


def read_UCASS_software_data(file_name: typing.Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    import re

    def ole2datetime(oledt):
        OLE_TIME_ZERO = pd.to_datetime('1899-12-30 00:00:00')
        return OLE_TIME_ZERO + pd.Timedelta(days=float(oledt))

    metadata = pd.read_table(file_name, skiprows=lambda x: x not in range(12))
    bin_lower_boundaries = np.array(metadata.iloc[8, 0].replace(',', '.').split(';')[1:],
                                    dtype=float)
    bin_upper_boundaries = np.array(metadata.iloc[9, 0].replace(',', '.').split(';')[1:],
                                    dtype=float)
    combined_boundaries = np.concatenate([bin_lower_boundaries,
                                          bin_upper_boundaries[~np.isin(bin_upper_boundaries,
                                                                        bin_lower_boundaries)]])
    try:
        df = pd.read_csv(file_name, header=11, sep=';')
        df.index = df['OADateTime'].map(lambda _: ole2datetime(_))
    except ValueError:
        df = pd.read_csv(file_name, header=11, sep=';', decimal=',')
        df.index = df['OADateTime'].map(lambda _: ole2datetime(_))
    df.drop('OADateTime', axis=1, inplace=True)
    i, col_lst = 0, []
    for col in df.columns:
        if re.compile('Bin[0-9][0-9]').match(col):
            col_lst.append(combined_boundaries[i+1])
            i += 1
        else:
            col_lst.append(col)
    df.columns = col_lst
    df = df.loc[~df.index.isnull()]
    df.sort_index(inplace=True, axis=0)

    return df
