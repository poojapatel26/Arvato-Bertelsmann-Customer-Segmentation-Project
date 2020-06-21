import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc,accuracy_score,confusion_matrix

def engineer_PRAEGENDE_JUGENDJAHRE(df):
    '''
    Engineer two new attributes from PRAEGENDE_JUGENDJAHRE: MOVEMENT and GENERATION_DECADE
    Dominating movement of person's youth (avantgarde vs. mainstream; east vs. west)
    
     
    Final encooding of PRAEGENDE_JUGENDJAHRE based on descriptions :
     
    converted all Movement into two parts and 1 and 2
    "MOVEMENT": 
    - 1: Mainstream
    - 2: Avantgarde
    
    converted all the generations into 4 to 9 based on generation 
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
    
    So, we created only two parts whether or not its rural neighbourhood.
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
   
    Most common building type within the PLZ8 region
    
    Converted into BAUMAX_BUSINESS column for region belong business or not
    and BAUMAX_Family column for how many family houses have
    
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


def check_value(x):
    '''check the values for missing value'''
    if type(x) == float:
        return x
    elif x == 'X' or (x == 'XX'):
        return np.nan
    else:
        return float(x)



def clean_data_ML(df1, test_data=True):
    '''
    Perform feature trimming, row dropping for a given DataFrame 
    
    Input:
        df (DataFrame)
        test_data (Bool): df is test data, so no rows should be dropped, default=False
    Output:
        cleaned_df (DataFrame): cleaned df DataFrame
    '''

    #drops columns with more than 20% of missing values
    print("Drop columns with more than 20% of missing values and Droping unnecessary columns")
    print("droping column EINGEFUEGT_AM and D19_LETZTER_KAUF_BRANCHE because it contain too many different items")
   
   
    
    df1.drop(['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4','EXTSEL992','KK_KUNDENTYP',
                     'RT_KEIN_ANREIZ','CJT_TYP_6','D19_VERSI_ONLINE_QUOTE_12','CJT_TYP_2','EINGEZOGENAM_HH_JAHR',
                     'D19_LOTTO','CJT_KATALOGNUTZER','VK_ZG11','UMFELD_ALT','RT_SCHNAEPPCHEN','AGER_TYP', 'ALTER_HH', 
                     'D19_BANKEN_ONLINE_QUOTE_12','D19_GESAMT_ONLINE_QUOTE_12', 'D19_KONSUMTYP','D19_VERSAND_ONLINE_QUOTE_12',
                     'GEBURTSJAHR','KBA05_BAUMAX','TITEL_KZ','D19_BANKEN_DATUM', 'D19_BANKEN_OFFLINE_DATUM',                                        'D19_BANKEN_ONLINE_DATUM', 
                     'D19_GESAMT_DATUM',  'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM','D19_TELKO_DATUM', 
                     'D19_TELKO_OFFLINE_DATUM','D19_TELKO_ONLINE_DATUM',  'D19_VERSAND_DATUM', 'D19_VERSAND_OFFLINE_DATUM', 
                     'D19_VERSAND_ONLINE_DATUM', 'D19_VERSI_DATUM', 'D19_VERSI_OFFLINE_DATUM','D19_VERSI_ONLINE_DATUM',
                     'CAMEO_DEU_2015',  'LP_FAMILIE_FEIN', 'LP_STATUS_FEIN',  'ANREDE_KZ', 'GREEN_AVANTGARDE',  'SOHO_KZ',
                     'VERS_TYP',  'LP_LEBENSPHASE_GROB','LP_LEBENSPHASE_FEIN','EINGEFUEGT_AM','D19_LETZTER_KAUF_BRANCHE',
                     'CAMEO_INTL_2015'],axis=1,inplace=True)

   
    print("creating a copy of dataframe")
    df=df1.copy()
    
    try:
        df.drop(['PRODUCT_GROUP','CUSTOMER_GROUP','ONLINE_PURCHASE'], axis=1,inplace=True)
    except:
        pass
    
    
   
    #O -> 0, W -> 1
    print("Re-encode OST_WEST_KZ attribute")
    df['OST_WEST_KZ'].replace(['O','W'], [0, 1], inplace=True)
    
    
    
    #engineer PRAEGENDE_JUGENDJAHRE
    print("Feature Engineer PRAEGENDE_JUGENDJAHRE")
    df = engineer_PRAEGENDE_JUGENDJAHRE(df)
    
    
    
    #engineer_WOHNLAGE
    print("Feature Engineer WOHNLAGE")
    df = engineer_WOHNLAGE(df)
    
    
    
    #engineer_PLZ8_BAUMAX
    print("Feature Engineer PLZ8_BAUMAX")
    df = engineer_PLZ8_BAUMAX(df)  
    
    
    
    #change object type of CAMEO_DEUG_2015 to numeric type
    print("Feature extracting CAMEO_DEUG_2015")  
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(lambda x: check_value(x))
    
   
    #remove columns with kba
    print("remove columns with start with kba")
    kba_cols = df.columns[df.columns.str.startswith('KBA05')]
    df.drop(list(kba_cols), axis='columns', inplace=True)
    
    
    #name of column of df that contains XX string
    for i in df.columns:
        df[i].astype('str').apply(lambda x: print(df[i].name) if x.startswith('XX') else 'pass')

    #imputing nan values 
    print("Imputing Nan values")
    imp=Imputer(missing_values=np.nan, strategy='most_frequent')
    df[df.columns] = imp.fit_transform(df)
    
    print("Counting Nan values",np.isnan(df).sum().sum())
    
    
    #fillimg Nan values with zero
    #print("filling Nan values with Zero")
    #df.fillna(0, inplace=True)
          
    return df

def evaluate_model(cv, X_test, y_test):
    """Draw ROC curve for the model
    Args:
        Classification Model
        X_test, y_test, Array-like
    return: ROC curve
    """
    y_pred = cv.predict_proba(X_test)[:,1]
    #print('\nBest Parameters:', cv.best_params_)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr) # compute area under the curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])
    



