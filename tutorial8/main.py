def main():

    import pandas as pd
    dataset = pd.read_csv('mushrooms.csv')

    y = dataset.iloc[:, 0].values
    selected_X = dataset.iloc[:, 1:3].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    encoded_y = LabelEncoder().fit_transform(y)
    encoded_X = OneHotEncoder().fit_transform(selected_X).toarray()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, encoded_y, test_size=0.33, random_state=0)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=0, solver = 'lbfgs', multi_class = 'ovr', max_iter = 100 )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Write your code here. Do not change any other parts of the code
    from sklearn.metrics import precision_score
    #uncomment the following two lines when you write the code for them to run
    print('precision score: '+ str(precision_score(y_test, y_pred)))
    return float(precision_score(y_test, y_pred))
    
main()  