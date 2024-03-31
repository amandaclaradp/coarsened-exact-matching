# Coarsened Exact Matching (CEM)

## Background
In causal inference, there is common matching method to find control group called Propensity Score Matching (PSM). But this method is slow because need iteration to find exact same characteristic between one with treatment and one without treatment. Same size treatment and control will be produced using PSM. Coarsened Exact Matching (CEM) is an alternative solution to find control group based on same pre-intervention characteristic with treatment group that has been coarsen into several bins/groups. As a result, the size of the control and treatment groups will differ.

## Example Usage
This is example use of cem analysis using healtcare stroke data to see influence of smoking status on stroke

### Import Library
```
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
```

### Data Preparation
```
df = pd.read_csv('healtcare_stroke_data.csv')
def one_hot_encode(df):
    """
    One-hot encodes all object data type columns of a Pandas DataFrame.
    """
    # Get the object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    # One-hot encode the object columns
    df = pd.get_dummies(df, columns=obj_cols)
    return df

df = one_hot_encode(df).fillna(0)
```
defining variable
```
df_model = df[['age','hypertension','heart_disease','bmi','stroke','gender_Male','smoking_status_smokes', 'avg_glucose_level']].reset_index()

treatment = 'smoking_status_smokes'
outcome = 'stroke'
features = {'age':5, 'avg_glucose_level':2, 'bmi':6,}

df_model.head()
```
in features variable define the number of bins that we want to form for every feature. Selecting the right number of bins is very important to get good result of matching
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/6a29cef4-813c-4ffd-809f-3ce5b6686168)

### Coarsening Feature
***Function for coarsening feature:***
```
def coarse(input_df,input_feature,treatment,outcome):
    df_in_coarse = input_df[features.keys()].copy().reset_index()
        
    for i in input_feature.keys():
        df_in_coarse[i] = pd.cut(df[i], bins=input_feature[i], right=False)
        
    df_all = pd.merge(input_df,df_in_coarse,on=['index'],how='left',suffixes=('_ori', '_bin'))
    return df_all
```

```
df_coarse = coarse(df_model,features,treatment,outcome)
df_coarse
```
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/abdd4060-5c02-4af0-a090-4735bedafd74)

### Distribution Plot Before Matching
***Function for making KDE Plot:***
```
def kde_plot(input_data, input_features, treatment, label_treatment={0: 'Non Takers', 1: 'Takers'}):
    df_viz = pd.melt(input_data, id_vars=treatment, value_name='value') 

    df_viz_no_outliers = {}
    for i in input_features:
        mask = df_viz['variable'] == i
        df_viz_mask = df_viz[mask].copy()

        # Filtering percentile 0.05 and 0.95 each variable
        lower_bound = df_viz_mask['value'].quantile(0.05)
        upper_bound = df_viz_mask['value'].quantile(0.95)
        no_outliers = (df_viz_mask['value'] >= lower_bound) & (df_viz_mask['value'] <= upper_bound)
        data_no_outliers = df_viz_mask[no_outliers]

        # Scaling relative to Max after filter percentile
        max_value = data_no_outliers['value'].max()
        data_no_outliers['scaled_value'] = data_no_outliers['value']/max_value
        
        # Colecting Pandas each Variable
        df_viz_no_outliers[i] = data_no_outliers


    df_viz_final = pd.concat(df_viz_no_outliers.values())
    mask = df_viz_final['variable'].isin(input_features)

    df_viz_final['Treatment'] = df_viz_final[treatment].apply(lambda x: label_treatment[x])

    # plot distribution
    sns.set_theme(style='white')
    sns.catplot(data=df_viz_final[mask], x="variable", y="scaled_value", hue="Treatment", kind='violin',
                   split=True, inner="quart", palette="muted", height=4, aspect=2, legend_out=False)
    plt.xlabel('')
    plt.ylabel('Scaled Value')
    plt.legend(loc='upper right', fontsize='small')
    plt.show()
```

```
feature_kde_bef = ['age','avg_glucose_level','bmi']
kde_plot(df,feature_kde_bef,treatment)
```
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/140e2366-16f1-4267-a3e3-a8f890482b48)

### Matching Treatment & Control 
***Function for matching process:***
```
def matching(input_df,input_feature,treatment,outcome):
    df = input_df.copy().drop(columns=[outcome,'index'], errors='ignore')
    
    # all possibilities of class combination
    signature = df.filter(like='_bin', axis=1)
    list_col_sig = list(signature.columns)
    df_signature = signature.groupby(list_col_sig)
    df_signature = df_signature.apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)

    # get total row for treatment and control that match each other
    count_rows = lambda df: df.groupby(list(signature.columns)).size().reset_index(name=f'total_row_match')
    label = df[treatment]==1
    
    match_treatment = count_rows(df[label])
    match_control = count_rows(df[~label])

    result_match = pd.merge(match_treatment, match_control, on=list_col_sig, how='left', suffixes=('_treatment', '_control'))

    result_match = result_match[(result_match['total_row_match_treatment'] != 0) & (result_match['total_row_match_control'] != 0)]
    
    # calculate weight ATT
    weight_t_att = 1
    weight_c_att = (result_match['total_row_match_treatment']/result_match['total_row_match_control'])*(sum(result_match['total_row_match_control'])/sum(result_match['total_row_match_treatment']))
    
    result_match['weight_t_att'] = weight_t_att
    result_match['weight_c_att'] = weight_c_att
    
    # calculate weight atc
    weight_t_atc = (result_match['total_row_match_control']/result_match['total_row_match_treatment'])*(sum(result_match['total_row_match_treatment'])/sum(result_match['total_row_match_control']))
    weight_c_atc = 1
    
    result_match['weight_t_atc'] = weight_t_atc
    result_match['weight_c_atc'] = weight_c_atc
    
    # calculate weight ate
    weight_t_ate = (result_match['total_row_match_control']+result_match['total_row_match_treatment'])/(result_match['total_row_match_treatment']*(1+(sum(result_match['total_row_match_control'])/sum(result_match['total_row_match_treatment']))))
    weight_c_ate = (result_match['total_row_match_control']+result_match['total_row_match_treatment'])/(result_match['total_row_match_control']*(1+(sum(result_match['total_row_match_treatment'])/sum(result_match['total_row_match_control']))))
    
    result_match['weight_t_ate'] = weight_t_ate
    result_match['weight_c_ate'] = weight_c_ate
    
    # join result match to preliminary data
    result_match_treatment = pd.merge(df[label],result_match,on=list_col_sig,how='inner').drop(columns=['total_row_match_treatment','total_row_match_control','weight_c_att','weight_c_atc','weight_c_ate'])
    result_match_treatment.rename(columns = {'weight_t_att':'weight','weight_t_atc':'weight_atc','weight_t_ate':'weight_ate'}, inplace = True)
    
    result_match_control = pd.merge(df[~label],result_match,on=list_col_sig,how='inner').drop(columns=['total_row_match_treatment','total_row_match_control','weight_t_att','weight_t_atc','weight_t_ate'])
    result_match_control.rename(columns = {'weight_c_att':'weight','weight_c_atc':'weight_atc','weight_c_ate':'weight_ate'}, inplace = True) 
    
    result_match_all = pd.concat([result_match_treatment, result_match_control], ignore_index=True)
    
    # add weighted feature
    feature_weighted = result_match_all.filter(like='_ori', axis=1).columns

    for i in feature_weighted:
        new_col_name = f"{i}_weighted"
        result_match_all[new_col_name] = result_match_all[i] * result_match_all['weight'] 
    
    return result_match_all
```

```
df_matched = matching(df_coarse,features.keys(),treatment,outcome)
df_matched
```
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/a89a2c8a-c61b-44dc-9fa2-7a6e15ca29de)

### Before and After Matching Evaluation
before matching
```
feature_ori = df_coarse.filter(like='_ori', axis=1).columns

for var in feature_ori:
    print(f"{var} | Before matching")
    display(df_coarse.groupby(treatment)[var].describe())
```
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/381b1f90-e62e-4dce-9dee-bb64fff1171b)

after matching
```
feature_weighted = df_matched.filter(like='_weighted', axis=1).columns

for var in feature_weighted:
    print(f"{var} | after matching")
    display(df_matched.groupby(treatment)[var].describe())
```
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/01db10e1-2b10-4013-bf19-ec5551b9d8df)
mean of every feature after matching and weighting process is almost same between treatment and control

**Standardized Mean Difference (SMD)**

SMD expresses the size of intervention effect relative to the variability. In this case, SMD used to evaluate data after matching (to show the intervention has no effect because there are similar characteristic between treatment and control group)

***Function for calculating and visualizing SMD:***
calculating SMD
```
def smd(input_df,feature,treatment):
    agg_dict = {}
    agg_dict.update({x : ['mean', 'std'] for x in feature})
    
    table_agg = input_df.groupby(treatment).agg(agg_dict).reset_index()
    
    
    smd_val = []
    for i in feature:
        mean_1 = table_agg[i].values[0,0]
        mean_2 = table_agg[i].values[1,0]
        std_1 = table_agg[i].values[0,1]
        std_2 = table_agg[i].values[1,1]

        smd = (mean_1 - mean_2)/np.sqrt((std_1**2 + std_2**2)/2)
        smd = abs(smd)

        smd_val.append(smd)
        
    df_smd = pd.DataFrame({'smd': smd_val}, index=feature).reset_index()
    df_smd.columns = ['features','smd']
    
    return df_smd
```
```
smd_before = smd(df_coarse,feature_ori,treatment)
smd_before['features'] = features.keys()

smd_after = smd(df_matched,feature_weighted,treatment)
smd_after['features'] = features.keys()
```
visualizing SMD
```
def smd_plot(smd_before,smd_after):
    df_smd_all = pd.merge(smd_before, smd_after, on='features', how='left')
    df_smd_all.columns = ['features', 'smd_before', 'smd_after'] 
    
    x = df_smd_all.index
    y1 = df_smd_all['smd_before']
    y2 = df_smd_all['smd_after']


    plt.style.use('bmh')
    plt.rc('axes', labelsize=9) 
    plt.plot(y1, x, 'ko', label='Bef. Match')
    plt.plot(y2, x, 'k+', label='Aft. Match')
    plt.yticks(ticks=x, labels=df_smd_all['features'])
    plt.xlabel('Standardized Mean Different')
    plt.legend()
```
```
smd_plot(smd_before,smd_after)
```
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/942c3ba3-7800-40a6-ad72-cd52dc5abc69)

### Distribution Plot After Matching
```
feature_kde_weighted = ['age_ori_weighted','avg_glucose_level_ori_weighted','bmi_ori_weighted']
kde_plot(df_matched,feature_kde_weighted,treatment,label_treatment={0: 'LCG-Alike', 1: 'Takers'})
```
![image](https://github.com/amandaclaradp/coarsened-exact-matching/assets/77821582/a10b52fc-3ee9-463b-bcb8-945b2f4c1737)

Although the size of takers and non-takers differs and their distribution looks different due to the multiplication of features by their weights, on average, they are comparable.





