import time
import warnings
import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class JungOptimizer:

    def __init__(self, num_needles, needle_mask, mri_image=None):
        self.num_actual_needles = num_needles
        self.needle_mask = needle_mask
        self.mri_image = mri_image
        self.index_points = np.array(np.where(needle_mask > 0)).transpose()
        self.physical_points = self._transform_index_points_to_physical(self.index_points)

    def _transform_index_points_to_physical(self, cs, keep_unique=True):
        physical_cs = []
        for index_point in cs:
            temp_point = (int(index_point[2]), int(index_point[1]), int(index_point[0]))
            physical_point = self.mri_image.TransformIndexToPhysicalPoint(temp_point)
            physical_cs.append(physical_point)
        physical_cs = np.array(physical_cs)
        if keep_unique:
            physical_cs = np.unique(physical_cs, axis=0)
        return physical_cs

    def fit_curves_on_needle_points(self, points_per_cluster, degree, n_of_curve_points):
        final_curve_points = []
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        for key in points_per_cluster.keys():
            temp_needle_points = points_per_cluster[key]
            poly_features = poly.fit_transform(temp_needle_points[:, 2].reshape(-1, 1))
            regressor_x = LinearRegression()
            regressor_y = LinearRegression()
            regressor_x.fit(poly_features, temp_needle_points[:, 0])
            regressor_y.fit(poly_features, temp_needle_points[:, 1])
            z_new = np.linspace(np.min(temp_needle_points[:, 2]), np.max(temp_needle_points[:, 2]), n_of_curve_points)
            poly_features = poly.fit_transform(z_new.reshape(-1, 1))
            x_new = regressor_x.predict(poly_features)
            y_new = regressor_y.predict(poly_features)

            current_curve_points = np.array([x_new, y_new, z_new]).transpose()
            final_curve_points.append(current_curve_points)

        final_curve_points = np.array(final_curve_points)
        return final_curve_points

    def _initialize_clusters(self, eps_distance=1):
        all_points_per_cluster = {key: [] for key in range(self.num_actual_needles)}
        issue_detected = False

        for slice_idx, img_slice in enumerate(self.needle_mask):
            if np.sum(img_slice) != 0:
                non_zero_of_slice = np.array(np.where(img_slice > 0)).transpose()
                if not non_zero_of_slice.shape[0] > 1:
                    break
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clustering = SpectralClustering(n_clusters=self.num_actual_needles, affinity='rbf', gamma=0.01,
                                                    assign_labels='cluster_qr', random_state=0).fit(non_zero_of_slice)

                if not any(all_points_per_cluster.values()):
                    for key in all_points_per_cluster:
                        temp_cluster_points = non_zero_of_slice[clustering.labels_ == key]
                        temp_cluster_points = np.array \
                            ([np.concatenate([np.array([slice_idx]), item]) for item in temp_cluster_points])
                        all_points_per_cluster[key].append(temp_cluster_points)
                else:
                    points_per_cluster = {key: np.array([np.concatenate([np.array([slice_idx]), item])
                                                         for item in non_zero_of_slice[clustering.labels_ == key]])
                                          for key in np.unique(clustering.labels_)}
                    means_per_cluster = np.array \
                        ([np.mean(points_per_cluster[key], axis=0) for key in np.unique(clustering.labels_)])
                    # previous_centroids = np.array([np.mean(item[-1], axis=0) for item in all_points_per_cluster.values()])
                    previous_centroids = np.array \
                        ([np.mean(item[-1], axis=0) if len(item[-1]) > 0 else np.array([10.000, 10.000, 10.000])
                                                   for item in all_points_per_cluster.values()])
                    # Calculate the 2D Euclidean distance between all previous centroids and all current centroids. Assign all the
                    # current centroids to their closest previous centroid. Check if the distance of clusters is getting smaller than
                    # eps_distance, and if yes terminate
                    kd_tree = KDTree(previous_centroids[:, 1:])
                    labels = kd_tree.query(means_per_cluster[:, 1:])[1]
                    current_means = {label: np.squeeze(means_per_cluster[labels == label]) for label in np.unique(labels)}
                    current_means = {key: np.mean(value, axis=0) if len(value.shape) > 1 else value for key, value in current_means.items()}
                    centroid_dists = pdist(np.array(list(current_means.values()))[:, 1:])
                    if (centroid_dists < eps_distance).any():
                        issue_detected = True
                        break
                    else:
                        for key in all_points_per_cluster:
                            if key in labels:
                                temp_points = np.concatenate \
                                    ([points_per_cluster[key2] for key2 in points_per_cluster if labels[key2] == key], axis=0)
                                all_points_per_cluster[key].append(temp_points)

        if issue_detected:
            print('    Early termination due to cluster merging.')
        return all_points_per_cluster, issue_detected

    def _dists_of_points_to_curves(self, needle_points):
        reference_points = self.physical_points
        needle_points_resh = needle_points.reshape(needle_points.shape[0] * needle_points.shape[1], 3)
        dists_of_points_to_curves = cdist(self.physical_points, needle_points_resh)
        dists_of_points_to_curves = dists_of_points_to_curves.reshape(self.physical_points.shape[0], needle_points.shape[0],
                                                                      needle_points.shape[1])
        dists_of_points_to_curves = np.min(dists_of_points_to_curves, axis=2)
        return dists_of_points_to_curves

    def assign_points_to_needles(self, needle_points):
        dists_of_points_to_curves = self._dists_of_points_to_curves(needle_points)
        needle_assignment = dists_of_points_to_curves.argmin(axis=1)
        loss = np.mean(dists_of_points_to_curves.min(axis=1))
        points_per_cluster = {key: self.physical_points[needle_assignment == key] for key in np.unique(needle_assignment)}
        return points_per_cluster, loss

    def optimize_needles(self, max_iterations, n_of_curve_points, eps_distance=3, loss_eps=1e-5, degree=3):
        start = time.time()
        initial_clusters, issue_detected = self._initialize_clusters(eps_distance)
        initial_clusters = {key: self._transform_index_points_to_physical(np.concatenate(initial_clusters[key])) for key in initial_clusters
                            if len(initial_clusters[key]) >= 5}
        initial_points = self.fit_curves_on_needle_points(initial_clusters, degree=degree, n_of_curve_points=n_of_curve_points)

        curve_points = initial_points
        if len(curve_points.shape) > 1:
            loss_values = []
            for i in range(max_iterations):
                points_per_cluster, loss = self.assign_points_to_needles(curve_points)
                if i > 0 and abs(loss_values[-1] - loss) < loss_eps:
                    break
                else:
                    loss_values.append(loss)
                    curve_points = self.fit_curves_on_needle_points(points_per_cluster, degree, n_of_curve_points=n_of_curve_points)

            end = time.time() - start
            print \
                (f"    Time: {time.strftime('%H:%M:%S', time.gmtime(end))}, Iterations: { i +1}, Degree: 3, Loss: {loss_values[-1]}")
        else:
            print('    Total failure! 0 needles found!')
        return curve_points