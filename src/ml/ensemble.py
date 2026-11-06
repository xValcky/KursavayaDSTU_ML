"""Ансамбли: MajorityVoteClassifier (из ens.py курса), Bagging, AdaBoost и набор моделей."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """Ансамбль голосованием большинства.

    vote='classlabel' — argmax по меткам, 'probability' — argmax суммы вероятностей.
    weights — важность каждого классификатора (None — равные).
    """

    def __init__(self, classifiers, vote="classlabel", weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ("probability", "classlabel"):
            raise ValueError(f"vote must be 'probability' or 'classlabel'; got (vote={self.vote!r})")
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f"Number of classifiers and weights must be equal; got "
                             f"{len(self.weights)} weights, {len(self.classifiers)} classifiers")
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = [clone(clf).fit(X, self.lablenc_.transform(y)) for clf in self.classifiers]
        return self

    def predict(self, X):
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights, minlength=len(self.classes_))),
                axis=1, arr=predictions)
        return self.lablenc_.inverse_transform(maj_vote)

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        return np.average(probas, axis=0, weights=self.weights)

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        out = self.named_classifiers.copy()
        for name, step in self.named_classifiers.items():
            for key, value in step.get_params(deep=True).items():
                out[f"{name}__{key}"] = value
        return out


def make_mlp(hidden=(64, 32), alpha=1e-3) -> Pipeline:
    """Классифицирующая нейросеть: 2 скрытых слоя ReLU, Adam, ранняя остановка."""
    return Pipeline([("sc", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                                           solver="adam", alpha=alpha, learning_rate_init=1e-3,
                                           max_iter=600, early_stopping=True,
                                           validation_fraction=0.15, n_iter_no_change=25,
                                           random_state=1))])


def model_zoo(mlp_params: dict | None = None) -> dict:
    mlp = make_mlp(**(mlp_params or {}))
    logreg = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=1))])
    tree = DecisionTreeClassifier(max_depth=6, criterion="entropy", random_state=0)
    knn = Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15))])
    return {
        "Логистическая регрессия": logreg,
        "Дерево решений": tree,
        "KNN": knn,
        "Нейросеть MLP": mlp,
        "Majority Vote (ансамбль)": MajorityVoteClassifier([logreg, tree, knn, mlp], vote="probability",
                                                           weights=[1, 1, 1, 2]),
        "Bagging (деревья)": BaggingClassifier(
            estimator=DecisionTreeClassifier(criterion="entropy", random_state=1),
            n_estimators=200, bootstrap=True, n_jobs=1, random_state=1),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=1),
            n_estimators=200, learning_rate=0.1, random_state=1),
    }
