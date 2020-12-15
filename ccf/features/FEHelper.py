# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:04:15 2017

@author: zhengji
"""
import pandas as pd
import numpy as np


class FEHelper:    
    
    """ NOTE
    Time groups: need to be finalized, using CAH or other methods !!!
    """
    
    @staticmethod
    def isContinuous(df_col, start, end, freq):
        """Check if a data column is continuous""" 
        dates = pd.date_range(start=start, end=end, freq=freq)
        for d in dates:
            if not (df_col == d).any():
                return False
        return True
    
    @staticmethod
    def Monthgroup(x): 
        if x in [3,4,5]: # mois faibles
            return(1)
        elif x in [1,2,6,9,10,11,12]: # mois un peu plus élevés
            return(2)
        elif x in [7,8]: # mois à forte influence
            return(3)
    @staticmethod
    def Daygroup(x):
        if x in [0]:
            return(1)
        elif x in [1,2,3,4]:
            return(2)
        elif x in [5]:
            return(3)
        elif x in [6]:
            return(4)
    @staticmethod
    def Hourgroup(x): 
        if x in [0,1,2,3,4,5,6,7,20,21,22,23]:
            return(1)
        elif x in [9,10,11,14,15,16,17]:
            return(2)
        elif x in [8,12,13,18,19]:
            return(3)

    @staticmethod
    def isRegion(x): # Corse: 2A,2B; Guadeloupe: 971; Guyane: 973; La Réunion: 974; Martinique:972 are not in scope
            if x in (['01','03','07','15','26','38','42','43','63','69','73','74']):
                return("Auvergne-Rhone-Alpes")
            elif x in (['21','25','39','58','70','71','89','90']):
                return("Bourgogne-Franche-Comte") 
            elif x in (['22','29','35','56']):
                return("Bretagne") 
            elif x in (['18','28','36','37','41','45']):
                return("Centre-Val de Loire") 
            elif x in (['08','10','51','52','54','55','57','67','68','88']):
                return("Grand Est") 
            elif x in (['02','59','60','62','80']):
                return("Hauts-de-France") 
            elif x in (['75','77','78','91','92','93','94','95']):
                return("Ile-de-France") 
            elif x in (['14','27','50','61','76']):
                return("Normandie") 
            elif x in (['16','17','19','23','24','33','40','47','64','79','86','87']):
                return("Nouvelle-Aquitaine") 
            elif x in (['09','11','12','30','31','32','34','46','48','65','66','81','82']):
                return("Occitanie")
            elif x in (['44','49','53','72','85']):
                return("Pays de la Loire")
            elif x in (['04','05','06','13','83','84']):
                return("Provence-Alpes-Cote d'Azur")
            else:
                return("Excluded")
                
    @staticmethod            
    def toFloat(value):
            try:
                return float(value)
            except ValueError:
                return np.nan
    
    @staticmethod
    def lholidays():
        """Check if a data column is continuous""" 
        new_year = pd.DataFrame({
          'holiday': 'new_year',
          'ds': pd.to_datetime(['2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01','2018-01-01'
          ,'2019-01-01','2020-01-01','2021-01-01','2022-01-01']),
        })

        lundi_paques = pd.DataFrame({
          'holiday': 'lundi_paques',
          'ds': pd.to_datetime(['2012-04-09', '2013-04-01', '2014-04-21', '2015-04-06', '2016-03-28', '2017-04-17','2018-04-02'
          ,'2019-04-22', '2020-04-13','2021-04-05','2022-04-18']),
        })

        travail = pd.DataFrame({
          'holiday': 'travail',
          'ds': pd.to_datetime(['2012-05-01', '2013-05-01', '2014-05-01', '2015-05-01', '2016-05-01', '2017-05-01','2018-05-01'
          ,'2019-05-01','2020-05-01','2021-05-01','2022-05-01']),
        })

        victoire_1945 = pd.DataFrame({
          'holiday': 'victoire_1945',
          'ds': pd.to_datetime(['2012-05-08', '2013-05-08', '2014-05-08', '2015-05-08', '2016-05-08', '2017-05-08','2018-05-08'
          ,'2019-05-08','2020-05-08','2021-05-08','2022-05-08']),
        })

        ascension = pd.DataFrame({
          'holiday': 'ascension',
          'ds': pd.to_datetime(['2012-05-17', '2013-05-09', '2014-05-29', '2015-05-14', '2016-05-05', '2017-05-25','2018-05-10'
          ,'2019-05-30','2020-05-21','2021-05-13','2022-05-26']),
        })

        pentecote = pd.DataFrame({
          'holiday': 'pentecote',
          'ds': pd.to_datetime(['2012-05-28', '2013-05-20', '2014-06-09', '2015-05-25', '2016-05-16', '2017-06-05','2018-05-21'
          ,'2019-06-10','2020-05-31','2021-05-23','2022-06-05']),
        })

        nationale = pd.DataFrame({
          'holiday': 'nationale',
          'ds': pd.to_datetime(['2012-07-14', '2013-07-14', '2014-07-14', '2015-07-14', '2016-07-14', '2017-07-14','2018-07-14'
          ,'2019-07-14','2020-07-14','2021-07-14','2022-07-14']),
        })
        assomption = pd.DataFrame({
          'holiday': 'assomption',
          'ds': pd.to_datetime(['2012-08-15', '2013-08-15', '2014-08-15', '2015-08-15', '2016-08-15', '2017-08-15','2018-08-15'
          ,'2019-08-15','2020-08-15','2021-08-15','2022-08-15']),
        })

        toussaint = pd.DataFrame({
          'holiday': 'toussaint',
          'ds': pd.to_datetime(['2012-11-01', '2013-11-01', '2014-11-01', '2015-11-01', '2016-11-01', '2017-11-01','2018-11-01'
          ,'2019-11-01','2020-11-01','2021-11-01','2022-11-01']),
        })

        armistice = pd.DataFrame({
          'holiday': 'armistice',
          'ds': pd.to_datetime(['2012-11-11', '2013-11-11', '2014-11-11', '2015-11-11', '2016-11-11', '2017-11-11','2018-11-11'
          ,'2019-11-11','2020-11-11','2021-11-11','2022-11-11']),
        })

        noel = pd.DataFrame({
          'holiday': 'noel',
          'ds': pd.to_datetime(['2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25','2018-12-25'
          ,'2019-12-25','2020-12-25','2021-12-25','2022-12-25']),
        })


        holidays = pd.concat((new_year,lundi_paques,travail,victoire_1945,ascension,pentecote,nationale,assomption,toussaint,armistice,noel ))
        return holidays


    @staticmethod
    def aftholidays():
        """Check if a data column is continuous""" 
        new_year = pd.DataFrame({'aftholiday': 'new_year',
                         'ds': pd.to_datetime(['2012-01-02',
                                               '2013-01-02',
                                               '2014-01-02',
                                               '2015-01-02',
                                               '2016-01-04',
                                               '2017-01-02',
                                               '2018-01-02',
                                               '2019-01-02',
                                               '2020-01-02',
                                               '2021-01-04',
                                               '2022-01-02']),
                         })

        lundi_paques = pd.DataFrame({'aftholiday': 'lundi_paques',
                             'ds': pd.to_datetime(['2012-04-10',
                                                   '2013-04-02',
                                                   '2014-04-22',
                                                   '2015-04-07',
                                                   '2016-03-29',
                                                   '2017-04-18',
                                                   '2018-04-03',
                                                   '2019-04-23',
                                                   '2020-04-14',
                                                   '2021-04-06',
                                                   '2022-04-19']),
                             })

        travail = pd.DataFrame({'aftholiday': 'travail',
                        'ds': pd.to_datetime(['2012-05-02',
                                              '2013-05-02',
                                              '2014-05-02',
                                              '2015-05-04',
                                              '2016-05-02',
                                              '2017-05-02',
                                              '2018-05-02',
                                              '2019-05-02',
                                              '2020-05-04',
                                              '2021-05-02',
                                              '2022-05-02']),
                        })

        victoire_1945 = pd.DataFrame({'aftholiday': 'victoire_1945',
                              'ds': pd.to_datetime(['2012-05-09',
                                                    '2013-05-10',
                                                    '2014-05-09',
                                                    '2015-05-11',
                                                    '2016-05-09',
                                                    '2017-05-09',
                                                    '2018-05-09',
                                                    '2019-05-09',
                                                    '2020-05-11',
                                                    '2021-05-10',
                                                    '2022-05-09']),
                              })

        ascension = pd.DataFrame({'aftholiday': 'ascension',
                          'ds': pd.to_datetime(['2012-05-18',
                                                '2013-05-10',
                                                '2014-05-30',
                                                '2015-05-15',
                                                '2016-05-06',
                                                '2017-05-26',
                                                '2018-05-11',
                                                '2019-05-31',
                                                '2020-05-22',
                                                '2021-05-14',
                                                '2022-05-27']),
                          })

        pentecote = pd.DataFrame({'aftholiday': 'pentecote',
                          'ds': pd.to_datetime(['2012-05-29',
                                                '2013-05-21',
                                                '2014-06-10',
                                                '2015-05-26',
                                                '2016-05-17',
                                                '2017-06-06',
                                                '2018-05-22',
                                                '2019-06-11',
                                                '2020-06-01',
                                                '2021-05-24',
                                                '2022-06-06']),
                          })

        nationale = pd.DataFrame({'aftholiday': 'nationale',
                          'ds': pd.to_datetime(['2012-07-15',
                                                '2013-07-15',
                                                '2014-07-15',
                                                '2015-07-15',
                                                '2016-07-15',
                                                '2017-07-17',
                                                '2018-07-15',
                                                '2019-07-15',
                                                '2020-07-15',
                                                '2021-07-15',
                                                '2022-07-15']),
                          })

        assomption = pd.DataFrame({'aftholiday': 'assomption',
                           'ds': pd.to_datetime(['2012-08-16',
                                                 '2013-08-16',
                                                 '2014-08-18',
                                                 '2015-08-16',
                                                 '2016-08-16',
                                                 '2017-08-16',
                                                 '2018-08-16',
                                                 '2019-08-16',
                                                 '2020-08-16',
                                                 '2021-08-16',
                                                 '2022-08-16']),
                           })

        toussaint = pd.DataFrame({'aftholiday': 'toussaint',
                          'ds': pd.to_datetime(['2012-11-02',
                                                '2013-11-04',
                                                '2014-11-02',
                                                '2015-11-02',
                                                '2016-11-02',
                                                '2017-11-02',
                                                '2018-11-02',
                                                '2019-11-04',
                                                '2020-11-02',
                                                '2021-11-02',
                                                '2022-11-02']),
                          })

        armistice = pd.DataFrame({'aftholiday': 'armistice',
                          'ds': pd.to_datetime(['2012-11-12',
                                                '2013-11-12',
                                                '2014-11-12',
                                                '2015-11-12',
                                                '2016-11-14',
                                                '2017-11-12',
                                                '2018-11-12',
                                                '2019-11-12',
                                                '2020-11-12',
                                                '2021-11-12',
                                                '2022-11-14']),
                          })

        noel = pd.DataFrame({'aftholiday': 'noel',
                     'ds': pd.to_datetime(['2012-12-26',
                                           '2013-12-26',
                                           '2014-12-26',
                                           '2015-12-28',
                                           '2016-12-26',
                                           '2017-12-26',
                                           '2018-12-26',
                                           '2019-12-26',
                                           '2020-12-28',
                                           '2021-12-26',
                                           '2022-12-26']),
                     })
        aftholidays = pd.concat((new_year,lundi_paques,travail,victoire_1945,ascension,pentecote,nationale,assomption,toussaint,armistice,noel))

        
        return aftholidays