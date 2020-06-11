import numpy as np
import pandas as pd


def plot_attribute_distribution(df, row_names, attribute, n):
    '''
    Plot Distribution of attribute n first rows from sorted DataFrame
    
    Input:
        df (DataFrame): Sorted dataset by given attribute
        row_names (str): name of clolumn with names
        attribute (str): name of attribute which distribution will be plotted
        n: number of rows that will be plotted
        
    Output:
        None
    '''
    ax = df[:n].plot(x = row_names ,y = attribute,  kind='barh', figsize=(5,10))
    ax.invert_yaxis()
    ax.set_xlabel(attribute, size='large')
    ax.set_ylabel(row_names, size='large');
    ax.set_title('Distribution of {}'.format(attribute), size='large')

def engineer_PRAEGENDE_JUGENDJAHRE(df):
    '''
    Engineer two new attributes from PRAEGENDE_JUGENDJAHRE: MOVEMENT and GENERATION_DECADE
    PRAEGENDE_JUGENDJAHRE initial encoding
    Dominating movement of person's youth (avantgarde vs. mainstream; east vs. west)
    - -1: unknown
    -  0: unknown
    -  1: 40s - war years (Mainstream, E+W)
    -  2: 40s - reconstruction years (Avantgarde, E+W)
    -  3: 50s - economic miracle (Mainstream, E+W)
    -  4: 50s - milk bar / Individualisation (Avantgarde, E+W)
    -  5: 60s - economic miracle (Mainstream, E+W)
    -  6: 60s - generation 68 / student protestors (Avantgarde, W)
    -  7: 60s - opponents to the building of the Wall (Avantgarde, E)
    -  8: 70s - family orientation (Mainstream, E+W)
    -  9: 70s - peace movement (Avantgarde, E+W)
    - 10: 80s - Generation Golf (Mainstream, W)
    - 11: 80s - ecological awareness (Avantgarde, W)
    - 12: 80s - FDJ / communist party youth organisation (Mainstream, E)
    - 13: 80s - Swords into ploughshares (Avantgarde, E)
    - 14: 90s - digital media kids (Mainstream, E+W)
    - 15: 90s - ecological awareness (Avantgarde, E+W)
    
Final encooding:
    
    "MOVEMENT": 
    - 1: Mainstream
    - 2: Avantgarde
    “GENERATION_DECADE”:
    - 4: 40s
    - 5: 50s
    - 6: 60s
    - 7: 70s
    - 8: 80s
    - 9: 90s
'''    
    

    #create new binary attribute MOVEMENT with values Avantgarde (0) vs Mainstream (1)
    df['MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE']
    df['MOVEMENT'].replace([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                           [np.nan,np.nan,1,2,1,2,1,2,2,1,2,1,2,1,2,1,2], inplace=True)

    #create new ordinal attribute GENERATION_DECADE with values 40s, 50s 60s ... encoded as 4, 5, 6 ...
    df['GENERATION_DECADE'] = df['PRAEGENDE_JUGENDJAHRE']
    df['GENERATION_DECADE'].replace([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                                    [np.nan,np.nan,4,4,5,5,6,6,6,7,7,8,8,8,8,9,9], inplace=True) 

    #delete 'PRAEGENDE_JUGENDJAHRE'
    df_new = df.drop(['PRAEGENDE_JUGENDJAHRE'], axis=1)
    
    return df_new



def engineer_WOHNLAGE(df):
    '''
    Engineer RURAL_NEIGHBORHOOD from WOHNLAGE attribute
    "WOHNLAGE" feature could be divided into “RURAL_NEIGHBORHOOD” and “QUALITY_NEIGHBORHOOD”.
    However, there are 24% of rural data that will have missing values in "QUALITY_NEIGHBORHOOD”
    feature, therefore only binary "RURAL_NEIGHBORHOOD" feature was created inplace of "WOHNLAGE"
    Initial encoding of WOHNLAGE:
    
    Neighborhood quality (or rural flag)
    - -1: unknown
    -  0: no score calculated
    -  1: very good neighborhood
    -  2: good neighborhood
    -  3: average neighborhood
    -  4: poor neighborhood
    -  5: very poor neighborhood
    -  7: rural neighborhood
    -  8: new building in rural neighborhood
    
    Final encooding:
    
    "RURAL_NEIGBORHOOD"
    - 0: Not Rural
    - 1: Rural
'''

    #create new binary attribute RURAL_NEIGHBORHOOD with values Rural (1) vs NotRural(0)
    df['RURAL_NEIGHBORHOOD'] = df['WOHNLAGE']
    df['RURAL_NEIGHBORHOOD'].replace([-1,0,1,2,3,4,5,7,8], [np.nan,np.nan,0,0,0,0,0,1,1], inplace=True)

    #delete 'WOHNLAGE'
    df_new = df.drop(['WOHNLAGE'], axis=1)

    return df_new


def engineer_PLZ8_BAUMAX(df):
    '''
    Engineer PLZ8_BAUMAX_BUSINESS and PLZ8_BAUMAX_FAMILY attributes from PLZ8_BAUMAX attribute
   
    PLZ8_BAUMAX initial encoding:
    Most common building type within the PLZ8 region
    - -1: unknown
    -  0: unknown
    -  1: mainly 1-2 family homes
    -  2: mainly 3-5 family homes
    -  3: mainly 6-10 family homes
    -  4: mainly 10+ family homes
    -  5: mainly business buildings
    
    Final encoding:
    “PLZ8_BAUMAX_BUSINESS”
    - 0: Not Business
    - 1: Business
    “PLZ8_BAUMAX_FAMILY”
    - 0: 0 families
    - 1: mainly 1-2 family homes
    - 2: mainly 3-5 family homes
    - 3: mainly 6-10 family homes
    - 4: mainly 10+ family homes
'''
    
    #create new binary attribute PLZ8_BAUMAX_BUSINESS with values Business (1) vs Not Business(0)
    df['PLZ8_BAUMAX_BUSINESS'] = df['PLZ8_BAUMAX']
    df['PLZ8_BAUMAX_BUSINESS'].replace([1,2,3,4,5], [0,0,0,0,1], inplace=True) 

    #create new ordinal attribute PLZ8_BAUMAX_FAMILY with from 1 to 4 encoded as in data dictionary
    df['PLZ8_BAUMAX_FAMILY'] = df['PLZ8_BAUMAX']
    df['PLZ8_BAUMAX_FAMILY'].replace([5], [0], inplace=True) 

    #delete 'PLZ8_BAUMAX'
    df_new = df.drop(['PLZ8_BAUMAX'], axis=1)  

    return df_new


def clean_data(df, test_data=False):
    '''
    Perform feature trimming, row dropping for a given DataFrame 
    
    Input:
        df (DataFrame)
        test_data (Bool): df is test data, so no rows should be dropped, default=False
    Output:
        cleaned_df (DataFrame): cleaned df DataFrame
    '''

    #drops columns with more than 30% of missing values
    print("Drop columns with more than 30% of missing values and Droping unnecessary columns")
    print("column EINGEFUEGT_AM and D19_LETZTER_KAUF_BRANCHE have too many different items")
    print("droping ID column from dataset")

    
    df =  df.drop(['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4','EXTSEL992','KK_KUNDENTYP',
                     'RT_KEIN_ANREIZ','CJT_TYP_6','D19_VERSI_ONLINE_QUOTE_12','CJT_TYP_2','EINGEZOGENAM_HH_JAHR',
                     'D19_LOTTO','CJT_KATALOGNUTZER','VK_ZG11','UMFELD_ALT','RT_SCHNAEPPCHEN','AGER_TYP', 'ALTER_HH', 
                     'D19_BANKEN_ONLINE_QUOTE_12','D19_GESAMT_ONLINE_QUOTE_12', 'D19_KONSUMTYP','D19_VERSAND_ONLINE_QUOTE_12',
                     'GEBURTSJAHR','KBA05_BAUMAX','TITEL_KZ','D19_BANKEN_DATUM', 'D19_BANKEN_OFFLINE_DATUM',                                        'D19_BANKEN_ONLINE_DATUM', 
                     'D19_GESAMT_DATUM',  'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM','D19_TELKO_DATUM', 
                     'D19_TELKO_OFFLINE_DATUM','D19_TELKO_ONLINE_DATUM',  'D19_VERSAND_DATUM', 'D19_VERSAND_OFFLINE_DATUM', 
                     'D19_VERSAND_ONLINE_DATUM', 'D19_VERSI_DATUM', 'D19_VERSI_OFFLINE_DATUM','D19_VERSI_ONLINE_DATUM',
                     'CAMEO_DEU_2015',  'LP_FAMILIE_FEIN', 'LP_STATUS_FEIN',  'ANREDE_KZ', 'GREEN_AVANTGARDE',  'SOHO_KZ',
                     'VERS_TYP',  'LP_LEBENSPHASE_GROB','LP_LEBENSPHASE_FEIN','EINGEFUEGT_AM','D19_LETZTER_KAUF_BRANCHE',
                     'LNR'],axis=1,inplace=True)

    
    #remove rows with more than 25 missing attributes if it not a testing data, skip this step if it is test data
    print("missing before")
    df['n_missing'] = df.isnull().sum(axis = 1)
    print("missing sum after ")
    
    try:
        df = df.drop(['PRODUCT_GROUP','CUSTOMER_GROUP','ONLINE_PURCHASE'], axis=1)
    except:
        pass
    
    if not test_data:
        print("Remove rows with more than 25 missing attributes")
        df = df[df["n_missing"]<= 25].drop("n_missing", axis=1)
    else:
        df = df.drop("n_missing", axis=1)
    
   
    #O -> 0, W -> 1
    print("Reencode OST_WEST_KZ attribute")
    df['OST_WEST_KZ'].replace(['O','W'], [0, 1], inplace=True)
    
    
    #engineer PRAEGENDE_JUGENDJAHRE
    print("Engineer PRAEGENDE_JUGENDJAHRE")
    df = engineer_PRAEGENDE_JUGENDJAHRE(df)
    
    #engineer_WOHNLAGE
    print("Engineer WOHNLAGE")
    df = engineer_WOHNLAGE(df)
    
    #engineer_PLZ8_BAUMAX
    print("Engineer PLZ8_BAUMAX")
    df = engineer_PLZ8_BAUMAX(df)  
    
    #change object type of CAMEO_DEUG_2015 to numeric type
    print("Feature extracting CAMEO_DEUG_2015")  
    df=df[~df['CAMEO_DEUG_2015'].isin(['X'])]
    df["CAMEO_DEUG_2015"] = pd.to_numeric(df["CAMEO_DEUG_2015"])
    
    #remove columns with kba
    print("remove columns with start with kba")
    kba_cols = df.columns[df.columns.str.startswith('KBA')]
    df.drop(list(kba_cols), axis='columns', inplace=True)
    
    #fillimg Nan values with zero
    print("filling Nan values with Zero")
    df.fillna(0, inplace=True)
        
    return df