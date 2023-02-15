# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:34:45 2022

@author: st5536
"""

import typing
import pandas as pd


def pine_quality_removal(campaign: str, pine_id: str, short_name: bool = False,
                         path: typing.Optional[str] = None):
    from pathlib import Path
    import pandas as pd
    import re

    pine_path = Path(r'\\IMKAAF-SRV1\agm-field\Instruments\PINE-04-01\PaCE22')
    pine_quality_path = pine_path.joinpath('Quality_Control')
    pine_quality_files = [_ for _ in pine_quality_path.glob('*flags.log')]

    flag_dct = {}
    for i, file in enumerate(pine_quality_files):
        if file.stat().st_size == 0:
            continue
        op_id_raw = re.search(r'op_id_\d+_', file.stem).group(0)
        op_id = int(re.search(r'\d+', op_id_raw).group(0))
        flag_df = pd.read_table(file, header=None)
        flag_df = flag_df.iloc[:, 0].str.split(',', expand=True)
        flag_df_0 = flag_df.iloc[:, 0].str.split(':', expand=True)
        flag_df_1 = flag_df.iloc[:, 1].str.split(':', expand=True)
        flag_df = pd.concat([flag_df_0, flag_df_1], axis=1, ignore_index=True)
        flag_df.rename({0: 'type',
                        1: 'run_id',
                        2: 'phase',
                        3: 'description'}, axis=1, inplace=True)
        flag_df['run_id'] = flag_df['run_id'].map(lambda x: int(re.search(r'\d+', x).group(0)))
        flag_dct[op_id] = flag_df

    run_id_remove = {}
    for key, value in flag_dct.items():
        run_id_remove[key] = value[value['type'] == 'ERROR']['run_id'].to_list()


def read_pine_data(campaign: str, pine_id: str, short_name: bool = False,
                   header: int = 8,
                   path: typing.Optional[str] = None,
                   logbook_name: typing.Optional[str] = None,
                   logbook_header: bool = False,
                   run_removal: typing.Optional[dict] = None) -> pd.DataFrame:
    """
    Read pine data from server.
    TODO: Add flag removal.

    Parameters
    ----------
    campaign : str
        Name of campaign.
    pine_id : str
        ID of PINE, i.e. PINE-04-01.
    short_name : bool, optional
        If the filename contains the short_name instead of the full pine_id,
        this value should be set to True. The default is False.
    header : int, optional
        Value at which line the header is. The default is 8.
    path : typing.Optional[str], optional
        If no data can be found, a specific path can be specified. The default is None.
    logbook_name : typing.Optional[str], optional
        Name of the logbook. The default is None.
    logbook_header : bool, optional
        Does the logbook have a header? The default is False.
    run_removal : typing.Optional[dict], optional
        Remove some runs after quality control. The default is None.

    Returns
    -------
    pine_data : pd.DataFrame
        Pine data for a specific campaign.

    """
    import pandas as pd

    pine_id_dict = {'PINE-04-01': 'PINE-401',
                    'PINE-04-02': 'PINE-402',
                    'PINE-04-03': 'PINE-403'}
    if short_name:
        pine_id_short = pine_id_dict[pine_id]
    else:
        pine_id_short = pine_id

    if not path:
        path = fr"\\IMKAAF-SRV1\MessPC\PINE\{pine_id}\{campaign}"

    path_operation = (path + fr"\pfo_{pine_id}_{campaign}.txt")

    try:
        operation = pd.read_csv(path_operation, sep=r'\t', engine='python',
                                parse_dates=['dto_start', 'dto_stop'])
    except FileNotFoundError:
        print(f'File {path_operation} not found. Checking other locations...')
        path = fr"\\IMKAAF-SRV1\agm-field\{campaign}\{pine_id}"
        path_operation = (path + fr"\pfo_{pine_id}_{campaign}.txt")
        try:
            operation = pd.read_csv(path_operation, sep=r'\t', engine='python',
                                    parse_dates=['dto_start', 'dto_stop'])
        except FileNotFoundError:
            path = input(
                f'File {path_operation} not found. '
                'Please provide the full path '
                'to the operation txt file, i.e.'
                fr"\\IMKAAF-SRV1\agm-field\Campaigns\{campaign}\{pine_id}")
            path_operation = (path + fr"\pfo_{pine_id}_{campaign}.txt")
            operation = pd.read_csv(path_operation, sep=r'\t', engine='python',
                                    parse_dates=['dto_start', 'dto_stop'])

    if not logbook_name:
        logbook = pd.read_excel(path + fr'\Logbook_{campaign}.xlsx')
    else:
        logbook = pd.read_excel(path + '\\' + logbook_name)
    if logbook_header:
        pass
    else:
        logbook.columns = logbook.iloc[0, :]
    logbook.drop(0, inplace=True)
    logbook = logbook[logbook['opreation type (#)'].notna()]
    op_ids = logbook['# operation'][logbook['opreation type (#)'].str.endswith(('(2)',
                                                                                '(3)'))].to_list()

    operation = operation[operation['op_id'].isin(op_ids)]

    pine_data_list = []
    for index in operation['op_id']:
        path_pfr = path + r"\raw_Data"
        if type(operation.loc[operation['op_id'] == index]['df_pfr'].to_numpy()[0]) != str:
            continue
        try:
            data_pfr = pd.read_csv(path_pfr + '\\'
                                   + operation.loc[
                                       operation['op_id'] == index]['df_pfr'].to_numpy()[0],
                                   sep=r'\t',
                                   engine='python',
                                   parse_dates=['time start', 'time expansion',
                                                'time refill', 'time end'])
        except FileNotFoundError:
            print('File: ', path_pfr + '\\'
                  + operation.loc[operation['op_id'] == index]['df_pfr'].to_numpy()[0],
                  'not found.')
            continue
        path_ice = (path + r"\L1_Data\exportdata\exportdata_ice")

        try:
            data_ice = pd.read_csv(
                path_ice + '\\' + f"{pine_id_short}_{campaign}_op_id"
                f"_{operation.loc[operation['op_id'] == index]['op_id'].to_numpy()[0]}_ice.txt",
                header=header, sep='\t', engine='python')
        except FileNotFoundError:
            print('File: ', path_ice + '\\' + f"{pine_id_short}_{campaign}_op_id"
                  f"_{operation.loc[operation['op_id'] == index]['op_id'].to_numpy()[0]}_ice.txt",
                  'not found.')
            continue

        if run_removal:
            for run_index, (run_start, run_stop) in run_removal.items():
                if index == run_index:
                    data_ice = data_ice.iloc[run_start:run_stop, :]

        pine_INP = pd.DataFrame({})
        pine_INP['T_min / K'] = [data_ice[data_ice['run_id'] == int(run)]['T_min'].to_numpy()[0]
                                 for run in data_pfr['RUN'].to_numpy()
                                 if run in data_ice['run_id'].to_numpy()]
        pine_INP['INP_cn / stdL-1'] = [(data_ice[data_ice['run_id'] == int(run)]['INP_cn_0']
                                        .to_numpy()[0])
                                       for run in data_pfr['RUN'].to_numpy()
                                       if run in data_ice['run_id'].to_numpy()]
        pine_INP.index = [data_pfr[data_pfr['RUN'] == int(run)]['time refill'].to_numpy()[0]
                          for run in data_pfr['RUN'].to_numpy()
                          if run in data_ice['run_id'].to_numpy()]
        pine_data_list.append(pine_INP)

    pine_data = pd.concat(pine_data_list)
    pine_data = pine_data[pine_data['INP_cn / stdL-1'] > 0]
    pine_data.sort_index(inplace=True)

    return pine_data
