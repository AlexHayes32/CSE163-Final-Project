'''
CSE 163 Final Project
Alex Hayes
Sedona Munguia
Creates a ML model, creates visualizations
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import export_graphviz
from Clean_Data import data


def user_input_data(data):
    '''
    Takes dataframe as a parameter. Prints a list of cities for
    that the user can choose from to create a ML model. Returns the
    city that is inputted.
    '''

    print(list(data.city.unique()))
    print('Enter desired city from list above')
    city = input()

    print('Predict price or volume?')
    print('Enter price/volume')
    model_type = input()

    return city, model_type


def setup_data(city, data):
    '''
    Takes in a user inputted city and the cleaned dataframe as
    parameters. Removes unnecessary columns for machine learning and
    groups by the city that is entered. Returns a dataframe that is
    ready for a ML model to be created on.
    '''

    data_ml = data[['AveragePrice', 'Total Volume', 'city', 'Total Bags',
                    'Small Bags', 'population', 'density', '4046', '4225',
                    '4770', 'Large Bags', 'XLarge Bags']]
    data_ml = data_ml[data_ml['city'] == city][0:169]
    data_ml = data_ml.drop(columns=['city'])

    return data_ml


def create_labels_features(data_ml, model_type):
    '''
    Takes in the ml dataframe and the model type as parameters.
    Creates the features and labels for the model, splits the data
    into training and testing data. Returns training and testing sets
    for both features and labels.
    '''

    if model_type == 'price':
        features = data_ml.loc[:, data_ml.columns != 'AveragePrice']
        labels = data_ml['AveragePrice']

    else:
        features = data_ml.loc[:, data_ml.columns != 'Total Volume']
        labels = data_ml['Total Volume']

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, train_size=.30788, shuffle=False)

    return [features_train, features_test, labels_train, labels_test]


def find_max_depth(lf_list):
    '''
    Takes in list of features and labels as parameters, runs through
    200 models to find the max_depth that produces the lowest MSE in
    the ml model. Returns max depth.
    '''

    max_depth_lst = []
    for max_depth in range(1, 201):
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(lf_list[0], lf_list[2])

        test_predictions = model.predict(lf_list[1])
        max_depth_lst.append(mean_squared_error(lf_list[3], test_predictions))

    return max_depth_lst.index(min(max_depth_lst)) + 1


def find_min_impurity_decrease(lf_list):
    '''
    Takes in list of features and labels as parameters, runs through
    200 models to find the min impurity decrease that produces the lowest
    MSE in the ml model. Returns min impurity decrease.
    '''

    min_impurity_lst = []
    for min_impurity in np.linspace(0, 1, 200):
        model = DecisionTreeRegressor(min_impurity_decrease=min_impurity)
        model.fit(lf_list[0], lf_list[2])

        test_predictions = model.predict(lf_list[1])
        min_impurity_lst.append(mean_squared_error
                                (lf_list[3], test_predictions))

    return min_impurity_lst.index(min(min_impurity_lst))


def find_max_leaf_nodes(lf_list):
    '''
    Takes in list of features and labels as parameters, runs through
    200 models to find the max leaf nodes that produces the lowest MSE in
    the ml model. Returns max leaf nodes.
    '''

    max_leaf_lst = []
    for max_leaf_nodes in range(2, 202):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
        model.fit(lf_list[0], lf_list[2])

        test_predictions = model.predict(lf_list[1])
        max_leaf_lst.append(mean_squared_error(lf_list[3], test_predictions))

    return max_leaf_lst.index(min(max_leaf_lst)) + 1


def find_min_samples_leaf(lf_list):
    '''
    Takes in list of features and labels as parameters, runs through
    200 models to find the min leaf sampls that produces the lowest MSE in
    the ml model. Returns min leaf samples.
    '''

    min_samples_lst = []
    for min_samples_leaf in range(1, 201):
        model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
        model.fit(lf_list[0], lf_list[2])

        test_predictions = model.predict(lf_list[1])
        min_samples_lst.append(mean_squared_error(lf_list[3],
                               test_predictions))

    return min_samples_lst.index(min(min_samples_lst)) + 1


def create_model(data_ml, max_depth, min_impurity, max_leaf_nodes,
                 min_leaf_samples, lf_list):
    '''
    Takes in the machine learning ready dataframe and the calculated
    hyperparameters as parameters. Creates a model from a decision
    tree regressor. Prints the MSE of the training and testing data sets,
    returns the model, training and testing predictions.
    '''

    model = DecisionTreeRegressor(max_depth=max_depth,
                                  min_impurity_decrease=min_impurity,
                                  max_leaf_nodes=max_leaf_nodes,
                                  min_samples_leaf=min_leaf_samples)
    model.fit(lf_list[0], lf_list[2])

    train_predictions = model.predict(lf_list[0])
    test_predictions = model.predict(lf_list[1])

    train_err = mean_squared_error(lf_list[2], train_predictions)
    test_err = mean_squared_error(lf_list[3], test_predictions)

    print('Training MSE:', train_err)
    print('Testing MSE:', test_err)

    return model, train_predictions, test_predictions


def plot_tree(model, lf_list):
    '''
    Takes the model, features, and labels as parameters, creates
    a decision tree based on these, plots the Graphviz source data
    as a pdf, saves the pdf in the plots folder.
    '''
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=lf_list[0].columns,
                               class_names=lf_list[2].unique(),
                               impurity=False,
                               filled=True, rounded=True,
                               special_characters=True)
    source_data = graphviz.Source(dot_data)

    source_data.render('plots/tree_model.gv', view=True)


def plot_feature_importances(model, model_type):
    '''
    Takes in the ML model and model type as parameters, plots a bar
    chart of the feature importances used in the ml model. Saves this
    fig in the plots folder as feature_importances.pdf.
    '''

    a = model.feature_importances_
    x = np.arange(10)
    plt.bar(x, a)
    plt.title('feature importance')

    if model_type == 'price':
        plt.xticks(x, ('Total Volume', 'Total Bags', 'Small Bags',
                       'population', 'density', '4046', '4225', '4770',
                       'Large Bags', 'XLarge Bags'), rotation='vertical')

    else:
        plt.xticks(x, ('Average Price', 'Total Bags', 'Small Bags',
                       'population', 'density', '4046', '4225', '4770',
                       'Large Bags', 'XLarge Bags'), rotation='vertical')

    plt.savefig('plots/feature_importances.pdf')


def plot_accuracy(model, model_type, data_ml,
                  city, train_predictions, test_predictions):
    '''
    Takes in the ml model, model type, and the ml dataframe
    as parameters, creates a plot visualizing the accuracy of
    the ml model for the city inputted. Saves this plot in the
    plots folder as plot_accuracy.pdf.
    '''

    if model_type == 'price':
        data_ml['Avg Price Prediction'] =\
            np.append(train_predictions, test_predictions)
        predict = data_ml['Avg Price Prediction']
        actual = data_ml['AveragePrice']

    else:
        data_ml['Total Volume Prediction'] = np.append(train_predictions,
                                                       test_predictions)
        predict = data_ml['Total Volume Prediction']
        actual = data_ml['Total Volume']

    fig, ax = plt.subplots()
    actual.plot(ax=ax, label='Actual')
    predict.plot(ax=ax, label='Predicted')
    ax.legend()
    ax.set_title(city + ' ' + model_type)

    plt.savefig('plots/plot_accuracy.pdf')


def main():
    '''
    Executes program.
    '''

    city, model_type = user_input_data(data)
    data_ml = setup_data(city, data)
    lf_list = create_labels_features(data_ml, model_type)
    max_depth = find_max_depth(lf_list)
    min_impurity = find_min_impurity_decrease(lf_list)
    max_leaf_nodes = find_max_leaf_nodes(lf_list)
    min_leaf_samples = find_min_samples_leaf(lf_list)
    model, train_predictions, test_predictions = create_model(
        data_ml, max_depth, min_impurity, max_leaf_nodes,
        min_leaf_samples, lf_list)

    plot_tree(model, lf_list)
    plot_feature_importances(model, model_type)
    plot_accuracy(model, model_type, data_ml, city,
                  train_predictions, test_predictions)


if __name__ == '__main__':
    main()
