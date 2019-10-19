import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor

#Drop features
def dropFeatures(file):
    df = pd.read_csv(file)
    instance = df['Instance']
    df = df.drop(['Instance','Wears Glasses','Hair Color','Body Height [cm]'],axis=1)
    return df,instance


#Fill missing values
def fillMissingValue(X):
    d = {}
    for c in list(X.columns):
        if X[c].dtypes == 'object':
            d[c] = 'NA'
        else:
            d[c] = X[c].mean()
    return X.fillna(value=d)
    

#Target Encoding
def calc_smooth_mean(df,cat_var,tgt_var,sm=300):
    dict_smooth = dict()
    #dict_gmean = dict()
    for col in cat_var:        
        #calculate the number of values and the mean of each group
        agg = df.groupby(col)[tgt_var].agg(['count','mean'])
        agg_count = agg['count']
        agg_mean = agg['mean']
        
        #calculate smooth_mean
        smooth = (agg_count * agg_mean + sm * agg_mean) / (agg_count + sm)
        df[col] =  df[col].map(smooth)
        dict_smooth[col] = smooth
    return df,dict_smooth



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--TrainData', help='Train data along with the path', type=str,default='TrainData.csv')
    parser.add_argument('--TestData', help='Test data along with the path', type=str,default='TestData.csv')
    parser.add_argument('--Output', help='Output file with path', type=str,default='Output.csv')
    args = parser.parse_args()

    if args.TrainData is None:
        print("Please specify the Training Data")
        exit(1)

    if args.TestData is None:
        print("Please specify the Test Data")
        exit(1)

    if args.Output is None:
        print("Please specify the Outputpath")
        exit(1)

    
    #Dataset after dropping features
    df_tr,tr_instance = dropFeatures(args.TrainData)
    df_ts,ts_instance = dropFeatures(args.TestData)
    
    #Dataset after filling missing Data
    df_tr = fillMissingValue(df_tr)
    df_ts = fillMissingValue(df_ts)
    #Call to Target Encoder
    df_tr,test_smooth = calc_smooth_mean(df_tr,[x for x in df_tr.columns if df_tr[x].dtypes == 'object'],
                   'Income in EUR')

    #Associate means of training to test data
    for c in [x for x in df_ts.columns if df_ts[x].dtypes == 'object']:
        df_ts.loc[:,c] = df_ts[c].map(test_smooth[c])
    
    #Dataset after filling missing Data
    df_ts = fillMissingValue(df_ts)
    
    #Train and Test Data    
    X_train = df_tr.iloc[:,:-1].values
    y_train =  df_tr.iloc[:,-1].values

    X_test = df_ts.iloc[:,:-1].values

    #Use RandomForestModel
    rfr = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=10)
    rfr.fit(X_train, y_train)
    
    y_pred = rfr.predict(X_test)


    '''
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state = 0)
    
    rfr = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=10)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    mean_error = np.sqrt(np.sum(((y_pred - y_test)**2))/len(y_test))
    print(mean_error)
    '''
    #Export the result to a csv file
    df_Income = pd.DataFrame(columns=['Instance','Income'])
    df_Income['Instance']=ts_instance
    df_Income['Income'] = y_pred
    df_Income.to_csv(args.Output)


if __name__ == '__main__':
    main()

    