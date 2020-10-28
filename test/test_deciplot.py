import unittest

from deciplot import DeciPlot2D
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def get_iris():
    iris = load_iris()
    return iris['data'], iris['target'], iris['feature_names'], iris['target_names']


class DeciPlot2DTest(unittest.TestCase):
    def test_get_couples(self):
        _ft_couples = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        X, y, feature_names, target_names = get_iris()
        dtree = DecisionTreeClassifier()
        dp2d = DeciPlot2D(X, y, feature_names, target_names, dtree)
        ft_couples = dp2d._get_feature_couples()

        self.assertEqual(ft_couples, _ft_couples)

    def test_get_feature_data(self):
        _ft_data = [5.1, 4.9, 4.7, 4.6, 5.]

        X, y, feature_names, target_names = get_iris()
        dtree = DecisionTreeClassifier()
        dp2d = DeciPlot2D(X, y, feature_names, target_names, dtree)
        ft_data = dp2d._get_feature_data(0)[:5]

        self.assertTrue((ft_data == _ft_data).all())

    def test_fit_predict(self):
        _pred = [2, 1, 0, 2, 0]

        X, y, feature_names, target_names = get_iris()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        dtree = DecisionTreeClassifier()
        dp2d = DeciPlot2D(X, y, feature_names, target_names, dtree)
        pred = dp2d._fit_predict(X_train, y_train, X_test[:5])

        self.assertTrue((pred == _pred).all())

    def test_not_a_classifier(self):
        _pred = [2, 1, 0, 2, 0]

        X, y, feature_names, target_names = get_iris()
        lr = LinearRegression()

        with self.assertRaises(Exception):
            dp2d = DeciPlot2D(X, y, feature_names, target_names, lr)

    def test_plot(self):
        X, y, feature_names, target_names = get_iris()
        dtree = DecisionTreeClassifier()
        try:
            dp2d = DeciPlot2D(X, y, feature_names, target_names, dtree)
            dp2d.plot(figsize=(10, 10))
        except Exception:
            self.fail('Exception during plotting')

