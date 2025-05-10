import time
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from lineartree import LinearTreeRegressor
from itertools import product

class LeonOptimizer:
    def __init__(self, needle_mask, mri_image, min_cluster_size=500, min_samples=15, xy_scale=3):
        self.needle_mask = needle_mask
        self.mri_image = mri_image
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.xy_scale = xy_scale
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

    def _cluster_needles(self):
        """Cluster needle voxels into individual needles using HDBSCAN."""
        # rescale x and y before clustering
        rescaled_coords = self.physical_points[:, :2] * self.xy_scale
        coords_rescaled = np.column_stack((rescaled_coords, self.physical_points[:, 2]))  # combine with z
        clustering = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples).fit_predict \
            (coords_rescaled)
        return clustering

    def _fit_curves_on_needle_points(self, points_per_cluster, degree, n_of_curve_points):
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

    @staticmethod
    def _project_reference_points(reference_points, line_points_top, line_points_bottom):
        """
        Calculate projected points from each needle point to each predicted line. Returns the projected points but also
        the projected alphas which are the scaled distances (between 0-1) from the top points to projected points along
        the predicted needle.
        Args:
            points_top:
            points_bottom:

        Returns:

        """
        direction = (line_points_bottom - line_points_top)
        direction_squared = np.sum(np.square(direction), axis=1, keepdims=True)
        num_lines = direction.shape[0]
        # offset alpha is the length of the vector starting from the origin and going to the projected point of the top
        # point on the direction vector
        offset_alpha = np.sum(np.multiply(line_points_top, direction), axis=1, keepdims=True) / direction_squared
        # reference_points_alpha is the length of the vector starting from the origin and going to the projected point
        # of the needle point to the direction vector
        reference_points_alpha = reference_points @ direction.T / direction_squared.T
        # this is the length of the vector we want to find (from top point to the projection of point on the line)
        proj_points_alphas = reference_points_alpha - offset_alpha.T
        # then we just find the projected points if we take the lengths and start from the top points of lines
        fl_proj_points_alphas = proj_points_alphas.reshape(-1, 1)
        num_point_line_pairs = fl_proj_points_alphas.shape[0]
        reptile = (int(num_point_line_pairs / num_lines), 1)
        direction_rsh = np.tile(direction, reptile)
        line_points_all = np.multiply(proj_points_alphas.reshape(-1, 1), direction_rsh)
        line_offsets = np.tile(line_points_top, reptile)
        # Shape: (l_idx + 3 * p_idx, # coords) - could reshape to (p_idx, l_idx, # coords)
        proj_points = line_points_all + line_offsets

        return proj_points

    @staticmethod
    def _dists_to_lines(reference_points, proj_points):
        temp_n_needles = (proj_points.shape[0] // reference_points.shape[0])
        proj_vecs = proj_points - np.repeat(reference_points, temp_n_needles, axis=0)
        dists = np.sqrt(np.sum(proj_vecs * proj_vecs, axis=1)).reshape(-1, temp_n_needles)
        return dists

    def _merge_needles(self, remaining_points, labels, consolidated_points, n_of_curve_points=50, eps=3):
        # given a set of clusters, check if there are clusters above and below the consolidated points. If there are clusters both
        # above and below, count the number of them. Fit lines on all top-bot combinations and keep the best
        # min(number_of_clusters_top, number_of_clusters_bot) lines
        points_per_cluster = {label: remaining_points[labels == label] for label in np.sort(np.unique(labels)) if label != -1}

        min_consolidated = consolidated_points[np.argmin(consolidated_points[:, 2])]
        max_consolidated = consolidated_points[np.argmax(consolidated_points[:, 2])]
        lower_and_upper_z = [(item[item[:, 2].argsort()][0], item[item[:, 2].argsort()][-1]) for item in points_per_cluster.values()]
        top_cluster_ids = [i for i, item in enumerate(lower_and_upper_z) if item[0][-1] >= max_consolidated[-1] - eps]
        bot_cluster_ids = [i for i, item in enumerate(lower_and_upper_z) if item[1][-1] <= min_consolidated[-1] + eps]
        final_pairs = []
        final_points_per_cluster = []

        if top_cluster_ids and bot_cluster_ids:
            all_combinations = list(product(top_cluster_ids, bot_cluster_ids))
            pairs_to_merge = []
            for combination in all_combinations:
                temp_cluster = {0: remaining_points[np.isin(labels, combination)]}
                temp_needle_points = self._fit_curves_on_needle_points(temp_cluster, degree=1, n_of_curve_points=n_of_curve_points)
                line_points_bottom = np.array([item[item[:, 2].argsort()][0] for item in temp_needle_points])
                line_points_top = np.array([item[item[:, 2].argsort()][-1] for item in temp_needle_points])
                proj_points = self._project_reference_points(temp_cluster[0], line_points_top, line_points_bottom)
                dists_to_lines = self._dists_to_lines(temp_cluster[0], proj_points)
                loss = np.mean(dists_to_lines)
                pairs_to_merge.append(combination + (loss,))

            # keep the best min(number_of_clusters_top, number_of_clusters_bot) lines
            final_pairs = [item[:-1] for item in sorted(pairs_to_merge, key=lambda x: x[2])[:min(len(top_cluster_ids, len(bot_cluster_ids)))]]

            for pair in final_pairs:
                # find the right order: lower -> upper, and make sure pairs of > 3 are legit
                ids_and_z_cs_of_bot_points = {j: lower_and_upper_z[j][-1] for j in pair}
                ids_and_z_cs_of_bot_points = dict(sorted(ids_and_z_cs_of_bot_points.items(), key=lambda item: item[1]))
                points_merged = np.concatenate([points_per_cluster[key] for key in ids_and_z_cs_of_bot_points], axis=0)
                final_points_per_cluster.append(points_merged)
                print(f'    {len(ids_and_z_cs_of_bot_points)} needles were merged')

        final_points_per_cluster = final_points_per_cluster + [value for key, value in points_per_cluster.items() if key not in sum(final_pairs, ())]
        return final_points_per_cluster

    def _separate_crossing_needles(self, labels, dist_limit, n_of_curve_points=50):
        """Separate needles that are crossing or touching."""
        separated_points = []
        unique_labels = np.unique(labels)

        for label_id in unique_labels:
            if label_id == -1:
                continue  # Ignore noise points

            cluster_points = self.physical_points[labels == label_id]

            # Check for large spread indicating crossing/touching needles
            consolidated = []
            spread_out = []
            z_values = np.unique(cluster_points[:, 2])
            for z in z_values:
                slice_points = cluster_points[cluster_points[:, 2] == z]
                centroid_of_slice = np.mean(slice_points, axis=0)
                max_dist_to_centroid = 0
                for temp_point in slice_points:
                    spread = np.linalg.norm(temp_point - centroid_of_slice)
                    if spread > max_dist_to_centroid:
                        max_dist_to_centroid = spread
                if max_dist_to_centroid > dist_limit:
                    spread_out.append(z)
                else:
                    consolidated.append(z)

            if spread_out:
                # Temporarily remove consolidated slices, re-cluster the remaining and merge them if needed
                spread_out_points = cluster_points[np.isin(cluster_points[:, 2], spread_out)]
                consolidated_points = cluster_points[np.isin(cluster_points[:, 2], consolidated)]
                rescaled_coords = spread_out_points[:, :2] * self.xy_scale
                coords_rescaled = np.column_stack((rescaled_coords, spread_out_points[:, 2]))  # combine with z
                if coords_rescaled.shape[0] >= self.min_samples:
                    new_labels = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples).fit_predict(coords_rescaled)
                    merged_points = self._merge_needles(spread_out_points, new_labels, consolidated_points, n_of_curve_points)
                    # assign labels to the consolidated points
                    temp_clusters = {key: value for key, value in enumerate(merged_points)}
                    temp_needle_points = self._fit_curves_on_needle_points(temp_clusters, degree=1, n_of_curve_points=n_of_curve_points)
                    if temp_needle_points.shape[0] != 0:
                        # TODO: assign consolidated points to clusters based on line distance and not point distance
                        line_points_bottom = np.array([item[item[:, 2].argsort()][0] for item in temp_needle_points])
                        line_points_top = np.array([item[item[:, 2].argsort()][-1] for item in temp_needle_points])
                        proj_points = self._project_reference_points(consolidated_points, line_points_top, line_points_bottom)
                        dists_to_lines = self._dists_to_lines(consolidated_points, proj_points)
                        needle_assignment = dists_to_lines.argmin(axis=1)
                        final_points = [np.concatenate([consolidated_points[needle_assignment == i], merged_points[i]]) for i in np.unique(needle_assignment)]
                        separated_points.extend(final_points)
                    else:
                        separated_points.append(cluster_points)
                else:
                    separated_points.append(cluster_points)
            else:
                separated_points.append(cluster_points)

        # Recombine separated points
        all_separated_points = np.vstack(separated_points) if separated_points else np.array([])
        new_labels = sum([[i] * len(item) for i, item in enumerate(separated_points)], [])
        return all_separated_points, new_labels

    # def _fit_polylines(self, coords, labels, n_sampled_points, mode='physical'):
    #     """Fit a polyline to each needle cluster."""
    #     needle_points = []
    #     unique_labels = np.unique(labels)

    #     for label_id in unique_labels:
    #         if label_id == -1:
    #             continue  # Ignore noise points

    #         cluster_points = coords[labels == label_id]
    #         cluster_points_with_mode = cluster_points[:, 2] if mode == 'physical' else cluster_points[:, 0]
    #         needle_line = []
    #         for z_value in np.sort(np.unique(cluster_points_with_mode)):
    #             z_cluster = cluster_points[cluster_points_with_mode == z_value]
    #             z_centroid = np.mean(z_cluster, axis=0)
    #             needle_line.append(z_centroid)
    #         needle_points.append(needle_line)

    #     needle_points = np.stack([generate_line_points(np.array(item), mode='n_points', n_points=n_sampled_points)
    #                                    for item in needle_points], axis=0)

    #     return needle_points

    def _fit_polylines(self, coords, labels, n_sampled_points, mode='physical'):

        final_curve_points = []
        unique_labels = np.unique(labels)

        for label_id in unique_labels:
            if label_id == -1:
                continue  # Ignore noise points

            cluster_points = coords[labels == label_id]
            if mode == 'physical':
                x_points_with_mode = cluster_points[:, 0]
                z_points_with_mode = cluster_points[:, 2]
            else:
                x_points_with_mode = cluster_points[:, 2]
                z_points_with_mode = cluster_points[:, 0]

            x_reg = LinearTreeRegressor(base_estimator=LinearRegression())
            y_reg = LinearTreeRegressor(base_estimator=LinearRegression())
            x_reg.fit(z_points_with_mode.reshape(-1 ,1), x_points_with_mode)
            y_reg.fit(z_points_with_mode.reshape(-1 ,1), cluster_points[:, 1])

            z_new = np.linspace(np.min(z_points_with_mode), np.max(z_points_with_mode), n_sampled_points)
            x_new = x_reg.predict(z_new.reshape(-1 ,1))
            y_new = y_reg.predict(z_new.reshape(-1 ,1))

            # x_pwlf = pwlf.PiecewiseLinFit(z_points_with_mode, x_points_with_mode)
            # y_pwlf = pwlf.PiecewiseLinFit(z_points_with_mode, cluster_points[:, 1])
            # res_x = x_pwlf.fitfast(10, pop=3)
            # res_y = y_pwlf.fitfast(10, pop=3)
            # z_new = np.linspace(np.min(z_points_with_mode), np.max(z_points_with_mode), n_sampled_points)
            # x_new = x_pwlf.predict(z_new)
            # y_new = y_pwlf.predict(z_new)

            if mode == 'physical':
                current_curve_points = np.array([x_new, y_new, z_new]).transpose()
            elif mode == 'index':
                current_curve_points = np.array([z_new, y_new, x_new]).transpose()
            final_curve_points.append(current_curve_points)

        final_curve_points = np.array(final_curve_points)
        return final_curve_points

    def optimize_needles(self, n_of_curve_points=50, dist_limit=3, print_results=True):
        start = time.time()
        labels = self._cluster_needles()
        coords, labels = self._separate_crossing_needles(labels, dist_limit=dist_limit, n_of_curve_points=n_of_curve_points)
        needle_points = self._fit_polylines(coords, labels, n_sampled_points=n_of_curve_points)
        if len(needle_points.shape) > 1:
            _, loss = self.assign_points_to_needles(needle_points)
            end = time.time() - start
            if print_results:
                print(f"    Time: {time.strftime('%H:%M:%S', time.gmtime(end))}, Loss: {loss}")
        else:
            if print_results:
                print('    Total failure! 0 needles found!')
        return needle_points