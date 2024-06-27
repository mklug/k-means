import numpy as np
import pandas as pd


class KMeans:
    '''
    K-means classifier mimicking a subset of the sklearn
    implementation and following their API.  
    '''

    def __init__(self,
                 n_clusters=8,
                 init='random',
                 random_state=None,
                 max_iter=300):
        '''
        Parameters:

        ``n_clusters`` : number of clusters (i.e., K).

        ``init`` : {'random', 'k-means++'} or a DataFrame
                   with index set 0,...,``n_clusters``.
                   Gives the initialization means or the method
                   of generating them.  'random' picks them
                   uniform randomly from the training data 
                   while 'k-means++' uses the k-means++
                   initialization algorithm.

        ``random_state`` : the random state used if ``init``
                           is a string.

        ``max_iter`` : maximum number of iterations.    
        '''

        self.n_clusters = n_clusters
        if type(init) == str:
            if init not in {'random', 'k-means++'}:
                raise Exception("""If not initialized with a DataFrame of 
                            means, ``init`` must be in {'random', 
                            'k-means++'}.""")
            self.init = init
        elif type(init) == pd.core.frame.DataFrame and \
            all(init.index == pd.RangeIndex(start=0,
                                            stop=self.n_clusters,
                                            step=1)):
            self.init = init
        else:
            raise Exception(
                """Invalid ``init`` input.  Must be a 
                DataFrame with index 0,...,``n_clusters``.""")

        self.random_state = random_state
        self.max_iter = max_iter

    def _is_close(df1, df2, rtol=1e-05, atol=1e-08):
        '''
        Given two DataFrames with the same index and columns,
        determines if all of the entries are close.
        '''
        if not all(df1.index == df2.index):
            raise Exception("DataFrames must have the same index.")
        if not all(df1.columns == df2.columns):
            raise Exception("DataFrames must have the same columns.")
        bools = np.array([np.isclose(df1.loc[i], df2.loc[i],
                                     rtol=rtol, atol=atol)
                          for i in df1.index])
        return bools.all()

    def fit(self, X_train):
        '''
        Takes training data and runs the Lloyd K-means algorithm.
        ``cluster_centers_`` and ``n_iter_`` are learned as attributes
        from the training process. 
        '''

        X_train = pd.DataFrame(X_train)

        if len(X_train) < self.n_clusters:
            raise Exception("""Not enough datapoints for the 
                            desired number of clusters.""")

        # initialize.
        if type(self.init) == str:
            if self.init == 'random':
                M = X_train.sample(self.n_clusters,
                                   random_state=self.random_state)
                M.reset_index(inplace=True, drop=True)

            elif self.init == 'k-means++':
                M = {0: X_train.sample(1,
                                       random_state=self.random_state).squeeze()}
                for i in range(1, self.n_clusters):
                    # Get the minimum distances from the rows of ``X_train``
                    # to the points in M.
                    D = X_train.apply(lambda x: min(np.linalg.norm(x-v)
                                                    for v in M.values()),
                                      axis=1)
                    M[i] = X_train.sample(1, weights=D**2,
                                          random_state=self.random_state).squeeze()
                M = pd.DataFrame(M.values(), index=M.keys())

        else:
            M = self.init
            if not all(M.columns == X_train.columns):
                raise Exception("""Initialized means and training data 
                                must have the same columns.""")

        # iterate.
        n_iter_ = 0
        for _ in range(self.max_iter):

            n_iter_ += 1
            # Series with index is the same as ``X_train``.
            # Entries are the indices of ``M`` the points are closest to.
            S = X_train.apply(lambda x: (M-x).apply(np.linalg.norm,
                                                    axis=1).idxmin(),
                              axis=1)
            M_new = pd.DataFrame([X_train[S == i].mean() for i in M.index],
                                 index=M.index)
            if KMeans._is_close(M, M_new):
                break
            M = M_new

        self.cluster_centers_ = M
        self.n_iter_ = n_iter_
        return self

    def transform(self, X_test):
        '''
        Transforms data into cluster-distance space.  
        Returns a DataFrame with the same index as the 
        input, with columns the indices of 0,...,``n_cluster``
        and with entries the distances of the input points
        to the various learned means.  
        '''
        M = self.cluster_centers_
        return X_test.apply(lambda x: (M-x).apply(np.linalg.norm,
                                                  axis=1),
                            axis=1)

    def predict(self, X_test):
        '''
        Returns a Series with the same index as the 
        input, with entries the indices in 0,...,``n_cluster``
        that are the closest cluster centers.  
        '''
        M = self.cluster_centers_
        X_test = pd.DataFrame(X_test)
        return X_test.apply(lambda x: (M-x).apply(np.linalg.norm,
                                                  axis=1).idxmin(),
                            axis=1)
