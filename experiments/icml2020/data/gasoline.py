"""
World Gasoline Demand Data, 18 OECD Countries, 19 target_c
Variables in the file are:
COUNTRY = name of country (Does not appear in the LIMDEP project file)
YEAR = year, 1960-1978
LGASPCAR = log of consumption per car
LINCOMEP = log of per capita income
LRPMG = log of real price of gasoline
LCARPCAP = log of per capita number of cars
See Baltagi (2001, p. 24) for analysis of these data. The article on which the analysis is based is Baltagi, B. and Griffin, J., "Gasolne Demand in the OECD: An Application of Pooling and Testing Procedures," European Economic Review, 22, 1983, pp. 117-137.  The data were downloaded from the website for Baltagi's text.

Data available at:
http://bcs.wiley.com/he-bcs/Books?action=resource&bcsId=4338&itemId=1118672321&resourceId=13452
"""
from .base import DatasetBase
import pandas as pd


class WorldGasolineDemand(DatasetBase):
    def __init__(self, path):
        self.UNNECESSARY_KEYS = ['YEAR']
        self.C_KEY = 'COUNTRY'
        self.Y_KEY = 'LGASPCAR'
        self.path = path
        self.train_size = 6


get_data = WorldGasolineDemand
