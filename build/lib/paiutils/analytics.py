"""
Author: Travis Hammond
Version: 12_10_2020
"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
from sklearn.manifold import (
    TSNE, Isomap, LocallyLinearEmbedding, MDS
)


class Analyzer:
    """Analyzer is a class used for manipulating and viewing
       classification datasets for analytical purposes. It can
       also be used for unclassified data by passing in the same
       value for y_data and the same label for all x_data.
    """

    def __init__(self, x_data, y_data, labels, label_colors=None):
        """Initializes the Analyzer with the dataset.
        params:
            x_data: A numpy ndarray
            y_data: A numpy ndarray, which is a onehot encoding or ndx
                    that corresponds to the label in labels
            labels: A list of strings, which are labels for the y_data
            label_colors: A list of list that contain 3 integers, which
                          represent a color of a label for plotting
        """
        assert len(x_data) == len(y_data)
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.labels = labels
        if self.y_data.ndim == 1:
            y_data_label_ndx = y_data
        elif self.y_data.ndim == 2:
            y_data_label_ndx = self.y_data.argmax(axis=1)
        else:
            raise NotImplementedError(
                'Cannot handle y_data that has more than 2 dimensions'
            )

        self.y_labels = np.array(labels)[y_data_label_ndx]
        if label_colors is None:
            self.colors = np.random.random(
                (len(self.labels), 3)
            )
        else:
            self.colors = label_colors
        self.y_colors = self.colors[y_data_label_ndx]

    def calculate_distribution_of_labels(self):
        """Calculates the number of samples in each label
        return: A dictionary with strings (labels) as
                keys and integers (number of samples) as values
        """
        groups = {label: 0 for label in self.labels}
        for ndx in range(len(self.y_labels)):
            groups[self.y_labels[ndx]] += 1
        return groups

    def create_label_ndx_groups(self):
        """Creates a dictionary with ndx of each group in each label.
        return: A dictionary with strings (labels) as keys
                and list of integers (indexes of x_data) as values
        """
        groups = {label: [] for label in self.labels}
        for ndx in range(len(self.y_labels)):
            groups[self.y_labels[ndx]].append(ndx)
        return groups

    def shrink_data(self, size_per_label, ndx_groups=None):
        """Creates an Analyzer with a dataset that has been shrunk
           by randomly choosing data from each group to get to the
           desired size of each group.
        params:
            size_per_label: A dictionary with labels as keys and sizes
                            as values, or an integer, which is the size
                            for all labels
            ndx_groups: A dictionary returned from create_label_ndx_groups
        return: An Analyzer
        """
        if ndx_groups is None:
            ndx_groups = self.create_label_ndx_groups()
        ndxs = []
        if isinstance(size_per_label, dict):
            for label, group in ndx_groups.items():
                if label in size_per_label:
                    size = size_per_label[label]
                    if size > 0:
                        ndxs.append(np.random.choice(
                            group, size, replace=False
                        ))
                else:
                    ndxs.append(group)
        else:
            for group in ndx_groups.values():
                ndxs.append(np.random.choice(
                    group, size_per_label, replace=False
                ))
        ndxs = np.hstack(ndxs)
        return Analyzer(self.x_data[ndxs], self.y_data[ndxs],
                        self.labels, label_colors=self.colors)

    def expand_data(self, size_per_label, ndx_groups=None):
        """Creates an Analyzer with a dataset that has been expanded
           by randomly choosing data from each group to get to the
           desired size of each group.
        params:
            size_per_label: A dictionary with labels as keys and sizes
                            as values, or an integer, which is the size
                            for all labels
            ndx_groups: A dictionary returned from create_label_ndx_groups
        return: An Analyzer
        """
        if ndx_groups is None:
            ndx_groups = self.create_label_ndx_groups()
        ndxs = []
        if isinstance(size_per_label, dict):
            for label, group in ndx_groups.items():
                if label in size_per_label:
                    size = size_per_label[label] - len(group)
                    if size > 0:
                        replace = size > len(group)
                        group = np.append(
                            group,
                            np.random.choice(group, size, replace=replace)
                        )
                ndxs.append(group)
        else:
            for group in ndx_groups.values():
                size = size_per_label - len(group)
                if size > 0:
                    replace = size > len(group)
                    group = np.append(
                        group,
                        np.random.choice(group, size, replace=replace)
                    )
                ndxs.append(group)
        ndxs = np.hstack(ndxs)
        return Analyzer(self.x_data[ndxs], self.y_data[ndxs],
                        self.labels, label_colors=self.colors)

    def plot(self, x, figsize=(8, 8)):
        """Plots x on a graph.
        params:
            x: A numpy ndarray of positonal points for x_data
            figsize: A tuple of 2 integers/floats, which are
                     width and height, respectively
        return: unmodified x
        """
        fig = plt.figure(figsize=figsize)
        dims = x.shape[1]
        if dims == 1:
            nx = np.squeeze(x)
            ax = fig.add_subplot(111)
            ax.scatter(nx, nx, c=self.y_colors)
            for ndx in range(len(self.labels)):
                ax.plot([], 'o', c=[self.colors[ndx]],
                        label=self.labels[ndx])
        elif dims == 2:
            ax = fig.add_subplot(111)
            ax.scatter(x[:, 0], x[:, 1], c=self.y_colors)
            for ndx in range(len(self.labels)):
                ax.plot([], 'o', c=self.colors[ndx],
                        label=self.labels[ndx])
        elif dims == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=self.y_colors)
            for ndx in range(len(self.labels)):
                ax.plot([[]], 'o', c=self.colors[ndx],
                        label=self.labels[ndx])
        fig.legend()
        plt.show()
        return x

    def boxplot(self, x, figsize=(8, 8), ndx_groups=None):
        """Creates a boxplot for each group of x
        params:
            x: A numpy ndarray of  1D positonal points for x_data
            figsize: A tuple of 2 integers/floats, which are
                     width and height, respectively
            ndx_groups: A dictionary returned from create_label_ndx_groups
        return: unmodified x
        """
        x = np.squeeze(x)
        if ndx_groups is None:
            ndx_groups = self.create_label_ndx_groups()
        labels = []
        data = []
        for label, ndxs in ndx_groups.items():
            labels.append(label)
            data.append(x[ndxs])

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        for label in labels:
            ndx = self.labels.index(label)
            ax.plot([], 's', c=self.colors[ndx],
                    label=self.labels[ndx])
            bp['boxes'][ndx].set_facecolor(self.colors[ndx])
        fig.legend()
        plt.show()
        return x

    def isomap(self, n_neighbors=5, n_components=3, eigen_solver='auto',
               tol=0, max_iter=None, path_method='auto',
               neighbors_algorithm='auto', n_jobs=None):
        """Creates an Isomap and fits x_data.
        params:
            n_neighbors: An integer, which is the number of neighbors
                         considered for each point
            n_components: An integer, which is the number of coordinates
                          for the manifold
            eigen_solver: A string ('auto', 'arpack', 'dense'),
                          which is solver for the problem
            tol: A float, which is the convergence tolerance for
                 eigen solvers (arpack, lobpcg)
            max_iter: An integer, which is the max number of iteration
                      for the arpack solver
            path_method: A string ('auto', 'FW', 'D'), which is the
                         algorthim used to find the shortest path
            neighbors_algorithm: A string ('auto', 'brute',
                                 'kd_tree', 'ball_tree'), which is the
                                 algorithm for nearest neighbors search
            n_jobs: An integer (-1 all), which is the number of parallel
                    jobs to run
        return: A numpy ndarray, which has a shape like
                (length of x_data, n_components)
        """
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components,
                        eigen_solver=eigen_solver, tol=tol, max_iter=max_iter,
                        path_method=path_method,
                        neighbors_algorithm=neighbors_algorithm, n_jobs=n_jobs)
        x_data = self.x_data.reshape(
            (self.x_data.shape[0], np.prod(self.x_data.shape[1:]))
        )
        return isomap.fit_transform(x_data)

    def locally_linear_embedding(self, n_neighbors=5, n_components=3, reg=1e-3,
                                 eigen_solver='auto', tol=1e-6, max_iter=100,
                                 method='standard', hessian_tol=1E-4,
                                 modified_tol=1E-12,
                                 neighbors_algorithm='auto',
                                 random_state=None,
                                 n_jobs=None):
        """Computes the locally linear embedding of x_data.
        params:
            n_neighbors: An integer, which is the number of neighbors
                         considered for each point
            n_components: An integer, which is the number of coordinates
                          for the manifold
            reg: A float, which is the regularization constant
            eigen_solver: A string ('auto', 'arpack', 'dense'),
                          which is solver for the problem
            tol: A float, which is the convergence tolerance for
                 eigen solvers (arpack)
            max_iter: An integer, which is the max number of iteration
                      for the arpack solver
            method: A string ('standard', 'hessian', 'modified', 'ltsa'),
                    which is the embedding algorithm
            hessian_tol: A float, which is the tolerance for Hessian method
            modified_tol: A float, which is the tolerance for LLE method
            neighbors_algorithm: A string ('auto', 'brute',
                                 'kd_tree', 'ball_tree'), which is the
                                 algorithm for nearest neighbors search
            random_state: An integer, which is a seed for random number
                          generator
            n_jobs: An integer (-1 all), which is the number of parallel
                    jobs to run
        return: A numpy ndarray, which has a shape like
                (length of x_data, n_components)
        """
        x_data = self.x_data.reshape(
            (self.x_data.shape[0], np.prod(self.x_data.shape[1:]))
        )
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                     n_components=n_components, reg=reg,
                                     eigen_solver=eigen_solver, tol=tol,
                                     max_iter=max_iter, method=method,
                                     hessian_tol=hessian_tol,
                                     modified_tol=modified_tol,
                                     neighbors_algorithm=neighbors_algorithm,
                                     random_state=random_state,
                                     n_jobs=n_jobs)
        return lle.fit_transform(x_data)

    def mds(self, n_components=3, metric=True, n_init=4, max_iter=300,
            verbose=0, eps=1e-3, random_state=None,
            dissimilarity='euclidean', n_jobs=None):
        """Creates a Multidimensional scaling and fits x_data.
        params:
            n_components: An integer, which is the number of dimensions
                          in which to immerse the dissimilarities
            metric: A boolean, which determines if metric MDS is performed
            n_init: An integer, which is the number of times the SMACOF
                    algortithm will be run with different initializations
            max_iter: An integer, which is the max number of iterations
                      of the SMACOF algorithm for a single run
            verbose: An integer, which determines the level of verbositity
            eps: A float, which is the relative tolerance with regard to stress
                 (determines when convergence is reached)
            random_state: An integer, which is a seed for random number
                          generator
            dissimilarity: A string ('euclidean', 'precomputed'), which
                           determines the measure to use
            n_jobs: An integer (-1 all), which is the number of parallel
                    jobs to run
        return: A numpy ndarray, which has a shape like
                (length of x_data, n_components)
        """
        x_data = self.x_data.reshape(
            (self.x_data.shape[0], np.prod(self.x_data.shape[1:]))
        )
        mds = MDS(n_components=n_components, metric=metric, n_init=n_init,
                  max_iter=max_iter, verbose=verbose, eps=eps, n_jobs=n_jobs,
                  random_state=random_state, dissimilarity=dissimilarity)
        return mds.fit_transform(x_data)

    def tsne(self, n_components=3, perplexity=30.0, early_exaggeration=12.0,
             learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
             min_grad_norm=1e-7, metric='euclidean', init='random', verbose=0,
             random_state=None, method='barnes_hut', angle=0.5, n_jobs=None):
        """Creates a t-distributed Stochastic Neighbor Embedding and fits x_data.
        params:
            n_components: An integer, which is the dimension of
                          embedded space
            perplexity: A float, which is related to the number of
                        nearest neigbors that are used in manifold learning
            early_exaggeration: A float, which controls how tight natural
                                clusters are embedded
            learning_rate: A float within 10.0-1000.0 (inclusive), which
                           higher makes data more like a 'ball', and lower
                           makes data cloudy with fewer outliers
            n_iter: An integer, which is the max number of iterations
                    for optimization
            n_iter_without_progress: An integer, which is the max number
                                     of iterations to continue without
                                     progress
            min_grad_norm: A float, which is the threshold for optimization
                           to continue
            metric: A string ('euclidean'), which determines the metric to use
                    for calculating distance between features arrays
            init: A string ('random', 'pca'), which determines how to
                  initalize the embedding
            verbose: An integer, which determines the level of verbositity
            random_state: An integer, which is a seed for random number
                          generator
            method: A string ('barnes_hut', 'exact'), which is the gradient
                    calculation algorithm
            angle: A float for barnes_hut, which is the determines the trade
                   off between speed and accuracy
            n_jobs: An integer (-1 all), which is the number of parallel
                    jobs to run
        return: A numpy ndarray, which has a shape like
                (length of x_data, n_components)
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate, n_iter=n_iter,
                    n_iter_without_progress=n_iter_without_progress,
                    min_grad_norm=min_grad_norm, metric=metric,
                    init=init, verbose=verbose, random_state=random_state,
                    method=method, angle=angle, n_jobs=n_jobs)
        x_data = self.x_data.reshape(
            (self.x_data.shape[0], np.prod(self.x_data.shape[1:]))
        )
        return tsne.fit_transform(x_data)
