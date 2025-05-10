import time
import warnings
import numpy as np
import networkx as nx

from itertools import combinations
from scipy.spatial import KDTree
from scipy.signal import oaconvolve
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression
from lineartree import LinearTreeRegressor

from utils import sample_points_linearly


class RobustAgainstNoiseOptimizer:

    def __init__(self, num_needles, needle_mask, mri_image=None, kernel_sizes=[(3, 5, 5)], axis=0):
        self.num_actual_needles = num_needles
        self.num_needles_found = num_needles
        self.needle_mask = needle_mask
        self.mri_image = mri_image
        self.kernel_sizes = kernel_sizes
        self.axis = axis
        self.index_points = np.array(np.where(needle_mask > 0)).transpose()
        self.noise_index_points = np.array([])
        self.noise_physical_points = np.array([])

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

    def _find_centroids_of_candidates(self, extracted_bot_points, extracted_top_points, eps_neighboor=3):
        high_clusters = []
        ids_that_belong_to_cluster = []
        for i in range(extracted_top_points.shape[0]):
            if i not in ids_that_belong_to_cluster:
                high_clusters.append([extracted_top_points[i]])
                ids_that_belong_to_cluster.append(i)
                for j in range(extracted_top_points.shape[0]):
                    if i != j and j not in ids_that_belong_to_cluster:
                        x_close = abs(extracted_top_points[i][0] - extracted_top_points[j][0]) <= eps_neighboor
                        y_close = abs(extracted_top_points[i][1] - extracted_top_points[j][1]) <= eps_neighboor
                        z_close = abs(extracted_top_points[i][2] - extracted_top_points[j][2]) <= eps_neighboor
                        if x_close and y_close and z_close:
                            ids_that_belong_to_cluster.append(j)
                            high_clusters[-1].append(extracted_top_points[j])
        final_top_points = [np.median(np.dstack([list(item2) for item2 in item1]), -1) for item1 in high_clusters]
        final_top_points = np.concatenate(final_top_points, axis=0) if final_top_points else []

        low_clusters = []
        ids_that_belong_to_cluster = []
        for i in range(extracted_bot_points.shape[0]):
            if i not in ids_that_belong_to_cluster:
                low_clusters.append([extracted_bot_points[i]])
                ids_that_belong_to_cluster.append(i)
                for j in range(extracted_bot_points.shape[0]):
                    if i != j and j not in ids_that_belong_to_cluster:
                        x_close = abs(extracted_bot_points[i][0] - extracted_bot_points[j][0]) <= eps_neighboor
                        y_close = abs(extracted_bot_points[i][1] - extracted_bot_points[j][1]) <= eps_neighboor
                        z_close = abs(extracted_bot_points[i][2] - extracted_bot_points[j][2]) <= eps_neighboor
                        if x_close and y_close and z_close:
                            ids_that_belong_to_cluster.append(j)
                            low_clusters[-1].append(extracted_bot_points[j])
        final_bot_points = [np.median(np.dstack([list(item2) for item2 in item1]), -1) for item1 in low_clusters]
        final_bot_points = np.concatenate(final_bot_points, axis=0) if final_bot_points else []
        return final_bot_points, final_top_points

    def _find_initial_points(self, eps_neighboor=3):
        kernel_sizes = self.kernel_sizes
        top_candidates = []
        bot_candidates = []

        for kernel_size in kernel_sizes:
            paddings = [(k, k) if idx == self.axis else ((k - 1) // 2, (k - 1) // 2) for (idx, k) in
                        enumerate(kernel_size)]
            example_img_padded = np.pad(self.needle_mask, paddings, constant_values=False)

            example_img_convolved = example_img_padded
            for axis, k in enumerate(kernel_size):
                kernel = np.ones([k if a == axis else 1 for (a, k) in enumerate(kernel_size)])
                example_img_convolved = oaconvolve(example_img_convolved, kernel, mode='valid', axes=axis)
            example_image_conv_mask = example_img_convolved >= 1
            h = self.needle_mask.shape[self.axis]
            o = kernel_size[self.axis]
            o = o + (o) // 2
            top_mask = np.take(example_image_conv_mask, range(0, h), axis=self.axis)
            bottom_mask = np.take(example_image_conv_mask, range(o, o + h), axis=self.axis)

            top_mask = self.needle_mask & ~top_mask
            bottom_mask = self.needle_mask & ~bottom_mask

            bot_candidates.append(np.array(np.where(top_mask > 0)).transpose())
            top_candidates.append(np.array(np.where(bottom_mask > 0)).transpose())

        extracted_bot_points = None
        all_bot_points = []
        all_top_points = []
        for item1, item2 in zip(bot_candidates, top_candidates):
            temp_bot_points, temp_top_points = self._find_centroids_of_candidates(item1, item2, eps_neighboor)
            all_bot_points.append(temp_bot_points)
            all_top_points.append(temp_top_points)
            if len(temp_bot_points) == len(temp_top_points) == self.num_actual_needles:
                extracted_bot_points = temp_bot_points
                extracted_top_points = temp_top_points
                break
        if extracted_bot_points is None:
            extracted_bot_points = min(all_bot_points, key=len)
            extracted_top_points = min(all_top_points, key=len)
        return extracted_bot_points, extracted_top_points

    def _fit_curves_on_needle_points(self, needle_assignment, degree, n_of_curve_points, return_outliers=False,
                                     data_type='physical', regressor_type='linear'):
        if data_type == 'physical':
            i1, i2, i3 = 0, 1, 2
        elif data_type == 'index':
            i1, i2, i3 = 2, 1, 0

        final_curve_points = []
        n_curves = len([item for item in np.unique(needle_assignment) if not np.isnan(item)])
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        for j in range(n_curves):
            if data_type == 'physical':
                temp_needle_points = self.physical_points[needle_assignment == j]
            elif data_type == 'index':
                temp_needle_points = self.index_points[needle_assignment == j]
            if temp_needle_points.shape[0] > 0:
                poly_features = poly.fit_transform(temp_needle_points[:, i3].reshape(-1, 1))
                regressor_x = RANSACRegressor(residual_threshold=5, min_samples=temp_needle_points.shape[0],
                                              max_trials=1) \
                    if regressor_type == 'ransac' else LinearRegression()
                regressor_y = RANSACRegressor(residual_threshold=5, min_samples=temp_needle_points.shape[0],
                                              max_trials=1) \
                    if regressor_type == 'ransac' else LinearRegression()
                regressor_x.fit(poly_features, temp_needle_points[:, i1])
                regressor_y.fit(poly_features, temp_needle_points[:, i2])
                z_new = np.linspace(np.min(temp_needle_points[:, i3]), np.max(temp_needle_points[:, i3]),
                                    n_of_curve_points)
                poly_features = poly.fit_transform(z_new.reshape(-1, 1))
                x_new = regressor_x.predict(poly_features)
                y_new = regressor_y.predict(poly_features)

                if data_type == 'physical':
                    current_curve_points = np.array([x_new, y_new, z_new]).transpose()
                elif data_type == 'index':
                    current_curve_points = np.array([z_new, y_new, x_new]).transpose()
                final_curve_points.append(current_curve_points)

        final_curve_points = np.array(final_curve_points)

        if not return_outliers:
            return final_curve_points
        else:
            final_curve_points = np.squeeze(final_curve_points)
            inlier_mask_x = regressor_x.inlier_mask_
            inlier_mask_y = regressor_y.inlier_mask_
            inliers = temp_needle_points[inlier_mask_x & inlier_mask_y]
            dists_of_points_to_curve = cdist(inliers, final_curve_points)
            dists_of_points_to_curve = np.min(dists_of_points_to_curve, axis=1)
            loss = np.mean(dists_of_points_to_curve)
            outlier_mask_x = np.logical_not(inlier_mask_x)
            outlier_mask_y = np.logical_not(inlier_mask_y)
            outliers = temp_needle_points[outlier_mask_x | outlier_mask_y]
            return final_curve_points, inliers, outliers, loss

    def _merge_needles(self, initial_points, n_of_curve_points=50):
        dists_of_points_to_curves = self._dists_of_points_to_curves(initial_points, mode='index')
        needle_assignment = dists_of_points_to_curves.argmin(axis=1)
        temp_points = [item for item in initial_points]
        all_combinations = list(combinations(range(len(temp_points)), 2))
        pairs_to_merge = []

        # find all combinations of 2 needles which should potentially be merged according to 3 conditions:
        # 1) the bottom point of a needle must be higher than the top point of another
        # 2) 2 needles shouls be in the same neighborhood in order to be merged
        # 3) if we fit ransac of 2 degree we shouldn't get outliers
        for combination in all_combinations:
            # check if there are pairs of possible split needles
            min_group_dist = np.min(cdist(temp_points[combination[0]][:, 1:], temp_points[combination[1]][:, 1:]))
            top_lower_than_bot = temp_points[combination[0]][-1][0] < temp_points[combination[1]][0][0]
            bot_higher_than_top = temp_points[combination[0]][0][0] > temp_points[combination[1]][-1][0]
            if (top_lower_than_bot or bot_higher_than_top) and min_group_dist <= 30:
                # merge needles and check if by fitting a 2nd order polynomial we get outliers
                temp_assignment = np.where(np.isin(needle_assignment, combination), 0, np.nan)
                if not len(np.unique(temp_assignment)) == 1:
                    try:
                        *_, outliers, loss = self._fit_curves_on_needle_points(temp_assignment, 2, n_of_curve_points,
                                                                               return_outliers=True,
                                                                               data_type='index',
                                                                               regressor_type='ransac')
                        if outliers.size == 0:
                            pairs_to_merge.append(combination + (loss,))
                    except ValueError:
                        continue

        # construct a graph of all possible combinations
        pairs_list = [tuple(pair[:-1]) for pair in pairs_to_merge]
        # Create a graph and add edges
        G = nx.Graph()
        G.add_edges_from(pairs_list)
        # Find all cliques in the graph
        cliques = list(nx.find_cliques(G))
        # Filter cliques to only those that are of size 3 or more
        final_pairs = [tuple(clique) for clique in cliques if len(clique) >= 3]
        # Convert final_pairs to a set to remove any duplicates that might have been added
        final_pairs = list(set(final_pairs + pairs_list))

        # calculate penalties for each combination
        pairs_penalty = []
        got_outliers = []
        for temp_pair in final_pairs:
            temp_assignment = np.where(np.isin(needle_assignment, temp_pair), 0, np.nan)
            try:
                temp_needle_points, _, outliers, _ = self._fit_curves_on_needle_points(temp_assignment, 2,
                                                                                       n_of_curve_points,
                                                                                       return_outliers=True,
                                                                                       data_type='index',
                                                                                       regressor_type='ransac')
                if outliers.size == 0:
                    got_outliers.append(False)
                else:
                    got_outliers.append(True)
                group_points = self.index_points[np.isin(needle_assignment, temp_pair)]
                dists_of_points_to_needle_points = cdist(group_points, temp_needle_points)
                closest_needle_points = np.unique(dists_of_points_to_needle_points.argmin(axis=1))
                len_penalty = len([item for item in range(temp_needle_points.shape[0])
                                   if item not in closest_needle_points])
                n_points = self.index_points[np.isin(needle_assignment, temp_pair)].shape[0]
                if n_points > 0:
                    pairs_penalty.append(((n_points + len_penalty) / n_points) - 1)
                else:
                    pairs_penalty.append(np.inf)
            except ValueError:
                got_outliers.append(True)
                pairs_penalty.append(np.inf)

        # filter out pairs if other pairs contain the same needles with smaller penalty
        needles_in_final_pairs = tuple(np.sort(np.unique(sum(final_pairs, ()))))
        final_pairs_filtered = []
        needles_checked = []
        for needle_id in needles_in_final_pairs:
            if needle_id not in needles_checked:
                pairs_with_same_needles = [item for item in final_pairs if needle_id in item and
                                           not any([item2 in needles_checked for item2 in item])]
                temp_needles = tuple(np.sort(np.unique(sum(pairs_with_same_needles, ()))))
                pairs_with_same_needles += [item for item in final_pairs if
                                            any([item2 in temp_needles for item2 in item]) and
                                            not any([item2 in needles_checked for item2 in item])]
                pairs_with_same_needles = list(set(pairs_with_same_needles))
                temp_penalties = [pairs_penalty[final_pairs.index(temp_pair)] for temp_pair in pairs_with_same_needles]
                temp_got_outliers = [got_outliers[final_pairs.index(temp_pair)] for temp_pair in
                                     pairs_with_same_needles]
                if len(pairs_with_same_needles) > 1:
                    filtered_pairs = [pairs_with_same_needles[i] for i in range(len(pairs_with_same_needles)) if
                                      not temp_got_outliers[i]]
                    if filtered_pairs:
                        filtered_penalties = [temp_penalties[i] for i in range(len(pairs_with_same_needles)) if
                                              not temp_got_outliers[i]]
                        best_pair = filtered_pairs[filtered_penalties.index(min(filtered_penalties))]
                        final_pairs_filtered.append(best_pair)
                        needles_checked.extend([item for item in best_pair])

                elif len(pairs_with_same_needles) == 0:
                    continue
                else:
                    final_pairs_filtered.append(pairs_with_same_needles[0])
                    needles_checked.extend(pairs_with_same_needles[0])

        initial_points_final = []
        for pair in final_pairs_filtered:
            # find the right order: lower -> upper, and make sure pairs of > 3 are legit
            ids_and_z_cs_of_bot_points = {j: temp_points[j][0][0] for j in pair}
            ids_and_z_cs_of_bot_points = dict(sorted(ids_and_z_cs_of_bot_points.items(), key=lambda item: item[1]))
            initial_points_merged = np.concatenate([initial_points[key] for key in ids_and_z_cs_of_bot_points], axis=0)
            initial_points_merged = sample_points_linearly(initial_points_merged, mode='n_points',
                                                         n_points=n_of_curve_points)
            initial_points_final.append(initial_points_merged)
            print(f'    {len(ids_and_z_cs_of_bot_points)} needles were merged')

        initial_points = np.stack(initial_points_final +
                                  [item for i, item in enumerate(initial_points) if
                                   i not in sum(final_pairs_filtered, ())], axis=0)

        # plt.close()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # points_df = pd.DataFrame(self.index_points, columns=['z','y','x'])
        # ax.plot(points_df['x'], points_df['y'], points_df['z'], 'o', c='Dodgerblue', lw=2, label='Points Found')

        # cmap = plt.get_cmap('rainbow', len(initial_points))
        # for i, item in enumerate(initial_points):
        #     temp_points = item.transpose()
        #     ax.plot(temp_points[2], temp_points[1], temp_points[0], 'o', c=cmap(i), lw=2, label=i)
        # plt.show()

        return initial_points

    def _remove_needles_until_num_of_needles_met(self, dists):
        dists_args_sorted = np.argsort(dists, axis=1)
        index_array = np.zeros((len(self.index_points), 2), int)
        index_array[:, 1] = 1
        num_needles_total = dists.shape[1]
        num_needles_desired = self.num_needles_found
        masked_needles_removed = np.full((num_needles_total), False)
        num_needles_left = num_needles_total
        penalties_per_needle = np.zeros(num_needles_total)

        while num_needles_left > num_needles_desired:
            indices_current = np.take_along_axis(dists_args_sorted, index_array[:, [0]], axis=1).reshape(-1)
            indices_next = np.take_along_axis(dists_args_sorted, index_array[:, [1]], axis=1).reshape(-1)
            # Accumulate penalties per point
            penalties_per_needle[:] = 0.0
            for p in range(len(self.index_points)):
                penalty_incurred = dists[p, indices_next[p]] - dists[p, indices_current[p]]
                penalties_per_needle[indices_current[p]] += penalty_incurred
            # Needle with the smallest penalty that has not yet been removed should be removed.
            penalties_per_needle_x = np.ma.masked_array(penalties_per_needle, masked_needles_removed)
            needle_to_remove = np.argmin(penalties_per_needle_x)
            # Reduce number of needles & marked as removed
            num_needles_left -= 1
            masked_needles_removed[needle_to_remove] = True
            # Update indexing matrix
            current_invalidated = indices_current == needle_to_remove
            next_invalidated = current_invalidated & (indices_next == needle_to_remove)
            # Move current to next if the current item was removed
            index_array[current_invalidated, 0] = index_array[current_invalidated, 1]
            # For each where the next element needs to be updated
            for p in next_invalidated.nonzero()[0]:
                # Seek to next valid element - sidenote - index will be out of bounds afterwards - may need to guard for this.
                index_array[p, 1] += 1
                while index_array[p, 1] < num_needles_total and masked_needles_removed[
                    dists_args_sorted[p, index_array[p, 1]]]: index_array[p, 1] += 1
        return masked_needles_removed

    def _modified_jung_initialization(self, bot_points, top_points, eps_distance=1):
        all_centroids_per_cluster = []
        for i, referece_points in enumerate([bot_points, top_points]):
            centroids_per_cluster = {key: [item] for key, item in enumerate(referece_points)}
            issue_detected = False

            needle_mask = self.needle_mask if i == 0 else self.needle_mask[::-1]
            for slice_idx, img_slice in enumerate(needle_mask):
                if np.sum(img_slice) != 0:
                    non_zero_of_slice = np.array(np.where(img_slice > 0)).transpose()
                    if non_zero_of_slice.shape[0] < 4:
                        continue
                    all_slice_points = np.array(
                        [np.concatenate([np.array([slice_idx]), item]) for item in non_zero_of_slice])

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clustering = SpectralClustering(n_clusters=len(referece_points), affinity='rbf', gamma=0.01,
                                                        assign_labels='cluster_qr', random_state=0).fit(
                            non_zero_of_slice)
                    # clustering = HDBSCAN(min_cluster_size=4, min_samples=4).fit(non_zero_of_slice)

                    points_per_cluster = {key: all_slice_points[clustering.labels_ == key] for key in
                                          np.unique(clustering.labels_)}
                    means_per_cluster = np.array(
                        [np.mean(points_per_cluster[key], axis=0) for key in np.unique(clustering.labels_)])
                    previous_centroids = np.array([item[-1] for item in centroids_per_cluster.values()])
                    kd_tree = KDTree(previous_centroids)
                    labels = kd_tree.query(means_per_cluster)[1]
                    current_means = {label: np.squeeze(means_per_cluster[labels == label]) for label in
                                     np.unique(labels)}
                    current_means = {key: np.mean(value, axis=0) if len(value.shape) > 1 else value for key, value in
                                     current_means.items()}
                    centroid_dists = pdist(np.array(list(current_means.values())))
                    if (centroid_dists < eps_distance).any():
                        issue_detected = True
                        break
                    else:
                        for label in np.unique(labels):
                            if np.linalg.norm(current_means[label] - centroids_per_cluster[label][-1]) > 0:
                                centroids_per_cluster[label].append(current_means[label])
            all_centroids_per_cluster.append(centroids_per_cluster)

        # decide which initialization was best based on the number of points
        n_points_bot = np.sum([len(item) for item in all_centroids_per_cluster[0].values()])
        n_points_top = np.sum([len(item) for item in all_centroids_per_cluster[1].values()])
        index_of_best = np.argmax([n_points_bot, n_points_top])

        if issue_detected:
            print('    Early termination due to cluster merging.')

        return all_centroids_per_cluster[index_of_best], issue_detected

    def _post_initialization(self, centroids_per_cluster, issue_detected, n_sampled_points=50):
        if sum(centroids_per_cluster.values(), []):
            initial_points = np.stack(
                [sample_points_linearly(np.array(centroids_per_cluster[key]), mode='n_points', n_points=n_sampled_points)
                 for key in centroids_per_cluster
                 if len(centroids_per_cluster[key]) > 1], axis=0)
            issue_detected = True if initial_points.shape[0] != self.num_actual_needles else issue_detected
            initial_points = self._merge_needles(initial_points, n_sampled_points)
            if initial_points.shape[0] < self.num_actual_needles:
                print(f'    Actual Needles: {self.num_actual_needles}, Needles found: {initial_points.shape[0]}')
                print(f'    Reducing number of optimized needles to {initial_points.shape[0]}')
                self.num_needles_found = initial_points.shape[0]
            elif initial_points.shape[0] > self.num_actual_needles:
                print(f'    Actual Needles: {self.num_actual_needles}, Needles found: {initial_points.shape[0]}')
                print(
                    f'    Ommiting {initial_points.shape[0] - self.num_actual_needles} needle(s) and corresponding points')
                dists = self._dists_of_points_to_curves(initial_points, mode='index')
                # scale dists
                # dists_median = np.median(dists.min(axis=1))
                scale_condition = dists > 5
                dists[scale_condition] = 5
                # dists[scale_condition] = dists[scale_condition] * (dists_median / dists[scale_condition])

                masked_needles_removed = self._remove_needles_until_num_of_needles_met(dists)
                needle_assignment = dists.argmin(axis=1)
                n_points_before = self.index_points.shape[0]
                self.noise_index_points = self.index_points[
                    np.isin(needle_assignment, np.argwhere(masked_needles_removed))]
                self.index_points = self.index_points[~np.isin(needle_assignment, np.argwhere(masked_needles_removed))]
                n_points_after = self.index_points.shape[0]
                print(f'    {n_points_before - n_points_after} points were marked as noise and removed')
                initial_points = initial_points[~masked_needles_removed]
        else:
            print('    No points found')
            initial_points = []

        return initial_points, issue_detected

    def _dists_of_points_to_curves(self, needle_points, mode):
        if mode == 'index':
            reference_points = self.index_points
        elif mode == 'physical':
            reference_points = self.physical_points
        needle_points_resh = needle_points.reshape(needle_points.shape[0] * needle_points.shape[1], 3)
        dists_of_points_to_curves = cdist(reference_points, needle_points_resh)
        dists_of_points_to_curves = dists_of_points_to_curves.reshape(reference_points.shape[0], needle_points.shape[0],
                                                                      needle_points.shape[1])
        dists_of_points_to_curves = np.min(dists_of_points_to_curves, axis=2)
        return dists_of_points_to_curves

    def _assign_points_to_needles(self, needle_points):
        dists_of_points_to_curves = self._dists_of_points_to_curves(needle_points, mode='physical')
        needle_assignment = dists_of_points_to_curves.argmin(axis=1)
        loss = np.mean(dists_of_points_to_curves.min(axis=1))
        return needle_assignment, loss

    # def optimize_needles(self, max_iterations, max_degree, n_of_curve_points, eps_neighboor=3, eps_distance=3, loss_eps=1e-5):
    #     start = time.time()
    #     bot_points, top_points = self._find_initial_points(eps_neighboor)
    #     centroids_per_cluster, issue_detected = self._modified_jung_initialization(bot_points, top_points, eps_distance)
    #     initial_points, _ = self._post_initialization(centroids_per_cluster, issue_detected, n_sampled_points=n_of_curve_points)
    #     initial_points = np.stack([self._transform_index_points_to_physical(item, keep_unique=False)
    #                                for item in initial_points], axis=0)
    #     self.physical_points = self._transform_index_points_to_physical(self.index_points)
    #     self.noise_physical_points = self._transform_index_points_to_physical(self.noise_index_points)
    #     needle_assignment, _ = self._assign_points_to_needles(initial_points)
    #     initial_points = self._fit_curves_on_needle_points(needle_assignment, degree=1, n_of_curve_points=n_of_curve_points)

    #     best_loss_per_degree = {key: {'loss': None, 'iteration': None, 'points': None} for key in range(1, max_degree+1)}
    #     for degree in range(1, max_degree+1):
    #         curve_points = initial_points
    #         loss_values = []
    #         for i in range(max_iterations):
    #             needle_assignment, loss = self._assign_points_to_needles(curve_points)
    #             if i > 0 and loss_values[-1] - loss < loss_eps:
    #                 break
    #             else:
    #                 loss_values.append(loss)
    #                 curve_points = self._fit_curves_on_needle_points(needle_assignment, degree, n_of_curve_points=n_of_curve_points)
    #         best_loss_per_degree[degree]['loss'] = loss_values[-1]
    #         best_loss_per_degree[degree]['iteration'] = i
    #         best_loss_per_degree[degree]['points'] = curve_points

    #     min_loss = np.min([best_loss_per_degree[d]['loss'] for d in best_loss_per_degree])
    #     for key in best_loss_per_degree:
    #         if best_loss_per_degree[key]['loss'] - min_loss < 1e-2:
    #             curve_points = best_loss_per_degree[key]['points']
    #             end = time.time() - start
    #             print(f"    Time: {time.strftime('%H:%M:%S', time.gmtime(end))}, Iterations: {best_loss_per_degree[degree]['iteration']}, "
    #                   f"Degree: {key}, Loss: {best_loss_per_degree[key]['loss']}")
    #             break

    #     return curve_points

    # def _fit_polylines_on_needle_points(self, needle_assignment, n_sampled_points, mode='physical'):

    #     needle_points = []
    #     unique_labels = np.unique(needle_assignment)

    #     for label_id in unique_labels:
    #         if label_id == -1:
    #             continue  # Ignore noise points

    #         if mode == 'physical':
    #             cluster_points = self.physical_points[needle_assignment == label_id]
    #             cluster_points_with_mode = cluster_points[:, 2]
    #         else:
    #             cluster_points = self.index_points[needle_assignment == label_id]
    #             cluster_points_with_mode = cluster_points[:, 0]
    #         needle_trajectory = []
    #         for z_value in np.sort(np.unique(cluster_points_with_mode)):
    #             z_cluster = cluster_points[cluster_points_with_mode == z_value]
    #             z_centroid = np.mean(z_cluster, axis=0)
    #             needle_trajectory.append(z_centroid)
    #         needle_points.append(needle_trajectory)

    #     needle_points = np.stack([generate_line_points(np.array(item), mode='n_points', n_points=n_sampled_points)
    #                                    for item in needle_points], axis=0)

    #     return needle_points

    def _fit_polylines_on_needle_points(self, needle_assignment, n_sampled_points, mode='physical'):
        # https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
        final_curve_points = []
        unique_labels = np.unique(needle_assignment)

        for label_id in unique_labels:
            if label_id == -1:
                continue  # Ignore noise points

            if mode == 'physical':
                cluster_points = self.physical_points[needle_assignment == label_id]
                x_points_with_mode = cluster_points[:, 0]
                z_points_with_mode = cluster_points[:, 2]
            else:
                cluster_points = self.index_points[needle_assignment == label_id]
                x_points_with_mode = cluster_points[:, 2]
                z_points_with_mode = cluster_points[:, 0]

            x_reg = LinearTreeRegressor(base_estimator=LinearRegression())
            y_reg = LinearTreeRegressor(base_estimator=LinearRegression())
            x_reg.fit(z_points_with_mode.reshape(-1, 1), x_points_with_mode)
            y_reg.fit(z_points_with_mode.reshape(-1, 1), cluster_points[:, 1])

            z_new = np.linspace(np.min(z_points_with_mode), np.max(z_points_with_mode), n_sampled_points)
            x_new = x_reg.predict(z_new.reshape(-1, 1))
            y_new = y_reg.predict(z_new.reshape(-1, 1))

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

    def optimize_needles(self, max_iterations, max_degree, n_of_curve_points, eps_neighboor=3, eps_distance=3,
                         loss_eps=1e-5,
                         fit_method='polynomial'):
        start = time.time()
        bot_points, top_points = self._find_initial_points(eps_neighboor)
        centroids_per_cluster, issue_detected = self._modified_jung_initialization(bot_points, top_points, eps_distance)
        initial_points, _ = self._post_initialization(centroids_per_cluster, issue_detected,
                                                      n_sampled_points=n_of_curve_points)
        initial_points = np.stack([self._transform_index_points_to_physical(item, keep_unique=False)
                                   for item in initial_points], axis=0)
        self.physical_points = self._transform_index_points_to_physical(self.index_points)
        self.noise_physical_points = self._transform_index_points_to_physical(self.noise_index_points)
        needle_assignment, _ = self._assign_points_to_needles(initial_points)

        if fit_method == 'polynomial':
            initial_points = self._fit_curves_on_needle_points(needle_assignment, degree=max_degree,
                                                               n_of_curve_points=n_of_curve_points)
        elif fit_method == 'polyline':
            initial_points = self._fit_polylines_on_needle_points(needle_assignment, n_of_curve_points)
        else:
            raise ValueError(f"Fit method '{fit_method}' is not implemented.")

        best_loss_per_degree = {key: {'loss': None, 'iteration': None, 'points': None} for key in
                                range(1, max_degree + 1)}
        for degree in range(1, max_degree + 1):
            curve_points = initial_points
            needle_assignment, loss = self._assign_points_to_needles(curve_points)
            loss_values = [loss]
            for i in range(max_iterations):
                if fit_method == 'polynomial':
                    curve_points = self._fit_curves_on_needle_points(needle_assignment, degree,
                                                                     n_of_curve_points=n_of_curve_points)
                elif fit_method == 'polyline':
                    curve_points = self._fit_polylines_on_needle_points(needle_assignment, n_of_curve_points)
                else:
                    raise ValueError(f"Fit method '{fit_method}' is not implemented.")
                needle_assignment, loss = self._assign_points_to_needles(curve_points)
                if abs(loss_values[-1] - loss) < loss_eps:
                    break
                else:
                    loss_values.append(loss)
            best_loss_per_degree[degree]['loss'] = loss_values[-1]
            best_loss_per_degree[degree]['iteration'] = i
            best_loss_per_degree[degree]['points'] = curve_points

        min_loss = np.min([best_loss_per_degree[d]['loss'] for d in best_loss_per_degree])
        for key in best_loss_per_degree:
            if best_loss_per_degree[key]['loss'] - min_loss < 1e-2:
                curve_points = best_loss_per_degree[key]['points']
                end = time.time() - start
                print(
                    f"    Time: {time.strftime('%H:%M:%S', time.gmtime(end))}, Iterations: {best_loss_per_degree[degree]['iteration']}, "
                    f"Degree: {key}, Loss: {best_loss_per_degree[key]['loss']}")
                break

        return curve_points
