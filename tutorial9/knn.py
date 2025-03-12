def main():
    from sklearn.datasets import load_iris
    X, _ = load_iris(return_X_y=True)

    sepal = X[:, :2]
    petal_length = X[:, 2]
    
    from sklearn.model_selection import train_test_split
    sepal_train, sepal_test, petal_length_train, petal_length_test = train_test_split(sepal, petal_length, test_size=0.33, random_state=0)

    # Write your code here. Do not change any other parts of the code
main()