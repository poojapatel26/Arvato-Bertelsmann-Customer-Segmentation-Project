import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc,accuracy_score,confusion_matrix

def check_value(x):
    '''
       converting string to float type
       replacing string 'X' and 'XX' with NaN 
    '''
    if type(x) == float:
        return x
    elif x == 'X' or (x == 'XX'):
        return np.nan
    else:
        return float(x)


def clean_data_ML(df1):
    '''
    Cleaning and performing feature extracting and engineering to the dataframe

    
    Input:
        df1 (DataFrame)
       
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
                     'CAMEO_INTL_2015','PRAEGENDE_JUGENDJAHRE','PLZ8_BAUMAX'],axis=1,inplace=True)

   
    print("creating a copy of dataframe")
    df=df1.copy()
    
    try:
        df.drop(['PRODUCT_GROUP','CUSTOMER_GROUP','ONLINE_PURCHASE'], axis=1,inplace=True)
    except:
        pass
    
    
   
    #replace O with 0 and W with 1
    print("Re-encode OST_WEST_KZ attribute")
    df['OST_WEST_KZ'].replace(['O','W'], [0, 1], inplace=True)
    
    #feature engineering Neighbourhood Column with three parts Rural(0), Not Rural(1) and Rural But Good Neighbourhood(2)    
    print("Feature engineering WOHLANG")
    df['TYPE_QUALITY_NEIGHBOURHOOD'] = df['WOHNLAGE']
    df['TYPE_QUALITY_NEIGHBOURHOOD'].replace([-1,0,1,2,3,4,5,7,8], [np.nan,np.nan,0,0,2,2,0,1,1], inplace=True)
    
    print("Droping WOHLANG column")
    #delete 'WOHNLAGE'
    df.drop(['WOHNLAGE'], axis=1,inplace=True)
    
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
          
    return df

def evaluate_model(cv, X_test, y_test):
    """Draw ROC curve for the model
    Input:
        Classification Model
        X_test, y_test, Array-like(model_classifer_obj)
    Output: 
        ROC curve
    """
    y_pred = cv.predict_proba(X_test)[:,1]
    fpr, tpr, th = roc_curve(y_test, y_pred)
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




