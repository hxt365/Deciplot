import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn


class DeciPlot:
    def plot(self):
        raise NotImplementedError


class DeciPlot2D(DeciPlot):
    def __init__(self, X, y, feature_names, target_names, classifier, step=0.02, marker='o', edgecolors='k'):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        self.classifier = classifier
        self.step = step
        self.marker = marker
        self.edgecolors = edgecolors

        if not sklearn.base.is_classifier(classifier):
            raise Exception('Model is not a classifier')

    def _get_feature_couples(self):
        couples = []
        n_features = len(self.feature_names)
        for ft1 in range(n_features):
            for ft2 in range(n_features):
                if ft1 < ft2:
                    couples.append((ft1, ft2))
        return couples

    def _get_feature_data(self, feature_name):
        return self.X[:, feature_name]

    def _fit_predict(self, x_train, y_train, x_test):
        self.classifier.fit(x_train, y_train)
        return self.classifier.predict(x_test)

    def _plot_data(self, axes, couples):
        for ax, (ft1, ft2) in zip(axes.ravel(), couples):
            data_ft1 = self._get_feature_data(ft1)
            data_ft2 = self._get_feature_data(ft2)

            for cls in np.unique(self.y):
                mask = self.y == cls
                ax.scatter(data_ft1[mask], data_ft2[mask],
                           marker=self.marker,
                           edgecolors=self.edgecolors,
                           label=self.target_names[cls])

            ax.set_xlabel(self.feature_names[ft1])
            ax.set_ylabel(self.feature_names[ft2])

    def _plot_decision_boundary(self, axes, couples):
        for ax, (ft1, ft2) in zip(axes.ravel(), couples):
            data_ft1 = self._get_feature_data(ft1)
            data_ft2 = self._get_feature_data(ft2)

            xx = np.arange(np.min(data_ft1) - 1, np.max(data_ft1) + 1, self.step)
            yy = np.arange(np.min(data_ft2) - 1, np.max(data_ft2) + 1, self.step)
            xgrid, ygrid = np.meshgrid(xx, yy)

            pred = self._fit_predict(x_train=np.c_[data_ft1, data_ft2],
                                     y_train=self.y,
                                     x_test=np.c_[xgrid.ravel(), ygrid.ravel()])
            pred = pred.reshape(xgrid.shape)

            ax.contourf(xx, yy, pred)

    def plot(self, fig_dim=None, figsize=None, legend=True):
        ft_couples = self._get_feature_couples()

        if fig_dim is None:
            fig_dim = math.ceil(len(ft_couples) / 2), 2
        fig, axes = plt.subplots(*fig_dim, figsize=figsize)

        self._plot_decision_boundary(axes, ft_couples)
        self._plot_data(axes, ft_couples)

        if legend:
            plt.legend()
