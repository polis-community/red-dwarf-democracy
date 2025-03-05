import numpy as np
from sklearn.cluster import KMeans

class KMeansPolis(KMeans):
    def __init__(self, n_clusters=2, tol_centers=0.01, max_iter=20, **kwargs):
        """
        Polis-like KMeans that stops when cluster centers move less than `tol_centers`.

        This modifies the default KMeans from a **stopping
        condition** (between iterations) that's inertia-based (`tol`), to direct
        cluster center movement comparison (`tol_centers`).

        Reference:
            <https://github.com/compdemocracy/polis/blob/56ed5c2618a0a372448d26bdaad3ae8a34c0ed33/math/src/polismath/math/clusters.clj#L68-L76>

        Parameters:
            n_clusters: Number of clusters.
            tol_centers: Minimum movement for centers to continue iterating.
            max_iter: Maximum iterations.
            kwargs: Other parameters for sklearn's KMeans.

        Returns:

        """
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, tol=0, **kwargs)
        self.tol_centers = tol_centers  # Custom stopping threshold for centers

    def _fit(self, X, y=None):
        """Overrides the main fit loop to check center movement."""
        # Initialize clusters using sklearn's normal method
        super()._check_params(X)
        self._n_threads = 1  # Ensure compatibility with sklearn threading

        best_inertia = None
        self.n_iter_ = 0
        old_centers = None

        for i in range(self.max_iter):
            self.n_iter_ = i + 1

            # Run one iteration of k-means (sklearn's internal logic)
            labels, new_centers, inertia = self._kmeans_single_lloyd(
                X, self.n_clusters, self.init, self.random_state, self.n_init,
                max_iter=1, verbose=self.verbose, tol=0, x_squared_norms=None)

            # Check stopping condition (center movement)
            if old_centers is not None:
                center_shifts = np.linalg.norm(new_centers - old_centers, axis=1)
                if np.all(center_shifts < self.tol_centers):
                    break  # Stop early, like `same-clustering?`

            old_centers = new_centers
            best_inertia = inertia

        # Store final model state
        self.cluster_centers_ = new_centers
        self.inertia_ = best_inertia
        self.labels_ = labels
        return self
