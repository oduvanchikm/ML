import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    if len(np.unique(feature_vector)) == 1:
        return None, None, None, None

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    unique_threshold_indices = np.where(np.diff(sorted_features) != 0)[0]
    thresholds = thresholds[unique_threshold_indices]

    if len(thresholds) == 0:
        return None, None, None, None

    left_counts = np.cumsum(sorted_targets[:-1])[unique_threshold_indices]
    left_totals = np.arange(1, len(sorted_features))[unique_threshold_indices]
    left_p1 = left_counts / left_totals
    left_p0 = 1 - left_p1

    right_counts = np.sum(sorted_targets) - left_counts
    right_totals = len(sorted_features) - left_totals
    right_p1 = right_counts / right_totals
    right_p0 = 1 - right_p1

    gini_left = 1 - left_p0 ** 2 - left_p1 ** 2
    gini_right = 1 - right_p0 ** 2 - right_p1 ** 2
    ginis = -(left_totals / len(sorted_features)) * gini_left - \
            (right_totals / len(sorted_features)) * gini_right

    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any([ft not in ("real", "categorical") for ft in feature_types]):
            raise ValueError("Unknown feature type in feature_types")

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._tree = {}

    def _fit_node(self, X, y, node, depth=0):
        if len(set(y)) == 1:
            node["type"] = "terminal"
            node["class"] = y[0]
            return

        if (self._max_depth is not None and depth >= self._max_depth) or len(y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(y).most_common(1)[0][0]
            return

        best_feature = None
        best_threshold = None
        best_gini = None
        best_split = None

        for i in range(X.shape[1]):
            feature_type = self._feature_types[i]
            feature_column = X[:, i]

            if feature_type == "real":
                feature_vector = feature_column
            elif feature_type == "categorical":
                counts = Counter(feature_column)
                clicks = Counter(feature_column[y == 1])
                ratios = {k: clicks.get(k, 0) / counts[k] for k in counts}
                sorted_categories = sorted(ratios.items(), key=lambda x: x[1])
                categories_map = {k: idx for idx, (k, _) in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[val] for val in feature_column])
            else:
                raise ValueError("Unsupported feature type")

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, y)
            if threshold is None:
                continue

            split = feature_vector < threshold
            if (np.sum(split) < self._min_samples_leaf or
                np.sum(~split) < self._min_samples_leaf):
                continue

            if best_gini is None or gini > best_gini:
                best_feature = i
                best_gini = gini
                best_split = split
                if feature_type == "real":
                    best_threshold = threshold
                else:
                    best_threshold = [k for k, v in categories_map.items() if v < threshold]

        if best_feature is None:
            node["type"] = "terminal"
            node["class"] = Counter(y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best_feature
        if self._feature_types[best_feature] == "real":
            node["threshold"] = best_threshold
        else:
            node["categories_split"] = best_threshold

        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(X[best_split], y[best_split], node["left_child"], depth + 1)
        self._fit_node(X[~best_split], y[~best_split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(np.array(X), np.array(y), self._tree)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_node(x, self._tree) for x in X])

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, "_" + key, value)
        return self
