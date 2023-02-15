# py_instruments

Implementation of various instrument readers in Python. Currently supported: PINE, APS, CPC3776, CPC3752, Fidas, UFCPC, CPC_AI, WS700, USMPS. The functions generally collect all the files inside a path, combine them into one dataframe and return that dataframe. The option to save the dataframe as a pickle is given as well.

It is recommended to install the instrument_reader via adding it to your path via conda-develop /path/to/module. This way you can easily import the functions into your own codes via

from py_instrument import read_data as io

or similar.
