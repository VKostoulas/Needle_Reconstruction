import time
import numpy as np

from utils import sample_points_linearly
from jung import JungOptimizer
from leon import LeonOptimizer
from mjung import RobustAgainstNoiseOptimizer


class MixedOptimizer(RobustAgainstNoiseOptimizer):

    def __init__(self, num_needles, needle_mask, mri_image=None, initialization='leon', kernel_sizes=[(3 ,5 ,5)], axis=0,
                 min_cluster_size=15, min_samples=5, xy_scale=3):
        super().__init__(num_needles, needle_mask, mri_image, kernel_sizes, axis)
        self.initialization = initialization

        self.num_actual_needles = num_needles
        self.num_needles_found = num_needles
        self.needle_mask = needle_mask
        self.mri_image = mri_image
        self.kernel_sizes = kernel_sizes
        self.axis = axis
        self.index_points = np.array(np.where(needle_mask > 0)).transpose()
        self.physical_points = self._transform_index_points_to_physical(self.index_points)
        self.noise_index_points = np.array([])
        self.noise_physical_points = np.array([])

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.xy_scale = xy_scale

    def _transform_physical_points_to_index(self, cs, keep_unique=True):
        index_cs = []
        for physical_point in cs:
            temp_point = (physical_point[0], physical_point[1], physical_point[2])
            index_point = self.mri_image.TransformPhysicalPointToIndex(temp_point)
            index_cs.append(tuple(reversed(index_point)))
        index_cs = np.array(index_cs)
        if keep_unique:
            index_cs = np.unique(index_cs, axis=0)
        return index_cs

    def _initialization(self, n_of_curve_points=50, eps_neighboor=3, eps_distance=1, dist_limit=3):
        if self.initialization == 'jung':
            jung_optimizer = JungOptimizer(self.num_actual_needles, self.needle_mask, self.mri_image)
            jung_clusters, jung_issue_detected = jung_optimizer._initialize_clusters(eps_distance)
            jung_points = \
                [sample_points_linearly(np.concatenate(jung_clusters[key]), mode='n_points', n_points=n_of_curve_points)
                for key in jung_clusters if len(jung_clusters[key]) > 1]
            if len(jung_points) > 0:
                jung_points = np.stack(jung_points, axis=0)
            else:
                jung_points = np.array([])
            return jung_points, jung_issue_detected
        if self.initialization == 'leon':
            leon_optimizer = LeonOptimizer(self.needle_mask, self.mri_image, self.min_cluster_size, self.min_samples,
                                           self.xy_scale)
            leon_points = leon_optimizer.optimize_needles(n_of_curve_points=n_of_curve_points, dist_limit=dist_limit,
                                                          print_results=False)
            leon_points = [self._transform_physical_points_to_index(item, keep_unique=True) for item in leon_points]
            leon_centroids_per_cluster = {key: list(leon_points[key]) for key in range(len(leon_points))}
            leon_points = [sample_points_linearly(np.array(leon_centroids_per_cluster[key]), mode='n_points',
                                                n_points=n_of_curve_points)
                           for key in leon_centroids_per_cluster if len(leon_centroids_per_cluster[key]) > 1]
            if len(leon_points) > 0:
                leon_points = np.stack(leon_points, axis=0)
            else:
                leon_points = np.array([])
            issue_detected = False
            return leon_points, issue_detected
        elif self.initialization == 'mjung':
            bot_points, top_points = self._find_initial_points(eps_neighboor)
            mjung_centroids_per_cluster, issue_detected = self._modified_jung_initialization(bot_points, top_points,
                                                                                             eps_distance)
            # print([np.array(mjung_centroids_per_cluster[key]).shape for key in mjung_centroids_per_cluster])
            mjung_points = np.stack([sample_points_linearly(np.array(mjung_centroids_per_cluster[key]), mode='n_points',
                                                          n_points=n_of_curve_points)
                                     for key in mjung_centroids_per_cluster if
                                     len(mjung_centroids_per_cluster[key]) > 1], axis=0)
            return mjung_points, issue_detected
        elif self.initialization == 'mixed':
            # leon
            leon_optimizer = LeonOptimizer(self.needle_mask, self.mri_image, self.min_cluster_size, self.min_samples,
                                           self.xy_scale)
            leon_points = leon_optimizer.optimize_needles(n_of_curve_points=n_of_curve_points, dist_limit=dist_limit,
                                                          print_results=False)
            leon_points = [self._transform_physical_points_to_index(item, keep_unique=True) for item in leon_points]
            leon_centroids_per_cluster = {key: list(leon_points[key]) for key in range(len(leon_points))}
            leon_points = [sample_points_linearly(np.array(leon_centroids_per_cluster[key]), mode='n_points',
                                                n_points=n_of_curve_points)
                           for key in leon_centroids_per_cluster if len(leon_centroids_per_cluster[key]) > 1]
            if len(leon_points) > 0:
                leon_points = np.stack(leon_points, axis=0)
            else:
                leon_points = np.array([])
            # jung
            jung_optimizer = JungOptimizer(self.num_actual_needles, self.needle_mask, self.mri_image)
            jung_clusters, jung_issue_detected = jung_optimizer._initialize_clusters(eps_distance)
            jung_points = [
                sample_points_linearly(np.concatenate(jung_clusters[key]), mode='n_points', n_points=n_of_curve_points)
                for key in jung_clusters if len(jung_clusters[key]) > 1]
            if len(jung_points) > 0:
                jung_points = np.stack(jung_points, axis=0)
            else:
                jung_points = np.array([])
            # jung_dists_of_points_to_curves = self._dists_of_points_to_curves(jung_points, mode='index')
            # needle_assignment = jung_dists_of_points_to_curves.argmin(axis=1)
            # jung_points = self._fit_polylines_on_needle_points(needle_assignment, n_of_curve_points, mode='index')
            # mjung
            bot_points, top_points = self._find_initial_points(eps_neighboor)
            mjung_centroids_per_cluster, mjung_issue_detected = self._modified_jung_initialization(bot_points,
                                                                                                   top_points,
                                                                                                   eps_distance)
            mjung_points = np.stack([sample_points_linearly(np.array(mjung_centroids_per_cluster[key]), mode='n_points',
                                                          n_points=n_of_curve_points)
                                     for key in mjung_centroids_per_cluster if
                                     len(mjung_centroids_per_cluster[key]) > 1], axis=0)
            # mjung_dists_of_points_to_curves = self._dists_of_points_to_curves(mjung_points, mode='index')
            # needle_assignment = mjung_dists_of_points_to_curves.argmin(axis=1)
            # mjung_points = self._fit_polylines_on_needle_points(needle_assignment, n_of_curve_points, mode='index')
            return ('jung', jung_points, jung_issue_detected), ('mjung', mjung_points, mjung_issue_detected), ('leon',
                                                                                                               leon_points,
                                                                                                               False)

    def optimize_needles(self, max_iterations, n_of_curve_points=50, eps_neighboor=3, eps_distance=3, loss_eps=1e-5,
                         dist_limit=3, min_max_degree=(1, 3), run_post_initialization=True, fit_method='polynomial'):
        start = time.time()
        opt_items = self._initialization(n_of_curve_points, eps_neighboor, eps_distance, dist_limit)

        if self.initialization != 'mixed':
            initial_points, issue_detected = opt_items
            if len(initial_points.shape) > 1:
                if run_post_initialization:
                    centroids_per_cluster = {key: list(initial_points[key]) for key in range(len(initial_points))}
                    initial_points, _ = self._post_initialization(centroids_per_cluster, issue_detected,
                                                                  n_sampled_points=n_of_curve_points)

                initial_points = np.stack([self._transform_index_points_to_physical(item, keep_unique=False)
                                           for item in initial_points], axis=0)
                self.physical_points = self._transform_index_points_to_physical(self.index_points)
                self.noise_physical_points = self._transform_index_points_to_physical(self.noise_index_points)

                if fit_method == 'polynomial':
                    best_loss_per_degree = {key: {'loss': None, 'iteration': None, 'points': None} for key in
                                            range(min_max_degree[0], min_max_degree[1] + 1)}
                    for degree in range(min_max_degree[0], min_max_degree[1] + 1):
                        curve_points = initial_points
                        needle_assignment, loss = self._assign_points_to_needles(curve_points)
                        loss_values = [loss]
                        for i in range(max_iterations):
                            curve_points = self._fit_curves_on_needle_points(needle_assignment, degree,
                                                                             n_of_curve_points=n_of_curve_points)
                            needle_assignment, loss = self._assign_points_to_needles(curve_points)
                            if abs(loss_values[-1] - loss) < loss_eps:
                                break
                            else:
                                loss_values.append(loss)
                        best_loss_per_degree[degree]['loss'] = loss_values[-1]
                        best_loss_per_degree[degree]['iteration'] = i + 1
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
                elif fit_method == 'polyline':
                    needle_assignment, loss = self._assign_points_to_needles(initial_points)
                    curve_points = self._fit_polylines_on_needle_points(needle_assignment, n_of_curve_points)
                    needle_assignment, loss = self._assign_points_to_needles(curve_points)
                    end = time.time() - start
                    print(f"    Time: {time.strftime('%H:%M:%S', time.gmtime(end))}, Loss: {loss}")
                else:
                    raise ValueError(f"Fit method '{fit_method}' is not implemented.")
            else:
                curve_points = np.array([])
                print('    Total failure! 0 needles found!')
            return curve_points

        else:
            opt_dict = {}
            for item in opt_items:
                opt_name, initial_points, issue_detected = item
                if len(initial_points.shape) > 1:
                    # print(opt_name, initial_points.shape)
                    if run_post_initialization:
                        centroids_per_cluster = {key: list(initial_points[key]) for key in range(len(initial_points))}
                        initial_points, _ = self._post_initialization(centroids_per_cluster, issue_detected,
                                                                      n_sampled_points=n_of_curve_points)

                    initial_points = np.stack([self._transform_index_points_to_physical(item, keep_unique=False)
                                               for item in initial_points], axis=0)
                    self.physical_points = self._transform_index_points_to_physical(self.index_points)
                    self.noise_physical_points = self._transform_index_points_to_physical(self.noise_index_points)
                    # print(opt_name, initial_points.shape)
                    if fit_method == 'polynomial':
                        best_loss_per_degree = {key: {'loss': None, 'iteration': None, 'points': None} for key in
                                                range(min_max_degree[0], min_max_degree[1] + 1)}
                        for degree in range(min_max_degree[0], min_max_degree[1] + 1):
                            curve_points = initial_points
                            needle_assignment, loss = self._assign_points_to_needles(curve_points)
                            loss_values = [loss]
                            for i in range(max_iterations):
                                curve_points = self._fit_curves_on_needle_points(needle_assignment, degree,
                                                                                 n_of_curve_points=n_of_curve_points)
                                needle_assignment, loss = self._assign_points_to_needles(curve_points)
                                if abs(loss_values[-1] - loss) < loss_eps:
                                    break
                                else:
                                    loss_values.append(loss)
                            best_loss_per_degree[degree]['loss'] = loss_values[-1]
                            best_loss_per_degree[degree]['iteration'] = i + 1
                            best_loss_per_degree[degree]['points'] = curve_points

                        min_loss = np.min([best_loss_per_degree[d]['loss'] for d in best_loss_per_degree])
                        for key in best_loss_per_degree:
                            if best_loss_per_degree[key]['loss'] - min_loss < 1e-2:
                                curve_points = best_loss_per_degree[key]['points']
                                break
                    elif fit_method == 'polyline':
                        needle_assignment, loss = self._assign_points_to_needles(initial_points)
                        curve_points = self._fit_polylines_on_needle_points(needle_assignment, n_of_curve_points)
                        needle_assignment, loss = self._assign_points_to_needles(curve_points)
                        # print((opt_name, f'pre-loss: {loss}'))
                    else:
                        raise ValueError(f"Fit method '{fit_method}' is not implemented.")

                    if len(self.noise_physical_points.shape) > 1:
                        self.physical_points = np.concatenate([self.physical_points, self.noise_physical_points],
                                                              axis=0)
                    # needle_assignment, loss = self._assign_points_to_needles(curve_points)
                    dists_of_points_to_curves = self._dists_of_points_to_curves(curve_points, mode='physical')
                    scale_condition = dists_of_points_to_curves > 5
                    dists_of_points_to_curves[scale_condition] = 5
                    loss = np.mean(dists_of_points_to_curves.min(axis=1))
                    opt_dict[opt_name] = {'points': curve_points, 'loss': loss}
                    self.index_points = np.array(np.where(self.needle_mask > 0)).transpose()
                    self.physical_points = self._transform_index_points_to_physical(self.index_points)
                    self.noise_index_points = np.array([])
                    self.noise_physical_points = np.array([])
                    self.num_needles_found = self.num_actual_needles

            # print([(key, opt_dict[key]['loss']) for key in opt_dict])
            min_loss_key = min(opt_dict, key=lambda k: opt_dict[k]['loss'])
            min_loss_dict = opt_dict[min_loss_key]
            end = time.time() - start
            print(f'    Best method was: {min_loss_key}')
            print(f"    Time: {time.strftime('%H:%M:%S', time.gmtime(end))}, Loss: {min_loss_dict['loss']}")
            return min_loss_dict['points']