# AVAILABLE STEPS

This document summarizes the main categories and options for each step in a typical scikit-learn pipeline. For each, you can find a link to the corresponding scikit-learn documentation to explore all available classes and parameters.

---

## 1. **Scaling / Normalization**

- 🔹 **Standard Scaler**
  - [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- 🔹 **Min-Max Scaler**
  - [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- 🔹 **Robust Scaler**
  - [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
- 🔹 **MaxAbs Scaler**
  - [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
- 🔹 **Normalizer (L1/L2)**
  - [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)
- 🔹 **Quantile Transformer**
  - [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
- 🔹 **Power Transformer (Yeo-Johnson, Box-Cox)**
  - [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

---

## 2. **Transformation (Feature Engineering)**

- 🔹 **Polynomial Features**
  - [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- 🔹 **FunctionTransformer**
  - [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
- 🔹 **KBins Discretizer**
  - [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
- 🔹 **Binarizer**
  - [Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)
- 🔹 **One-Hot, Ordinal, Label Encoding**
  - [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
  - [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
  - [LabelEncoder (for targets)](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- 🔹 **Target / Frequency Encoder**
  - Custom or via [CategoryEncoders](https://contrib.scikit-learn.org/category_encoders/)

---

## 3. **Feature Reduction / Selection**

### 🔹 **Dimensionality Reduction**
- [PCA (Principal Component Analysis)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [NMF (Non-Negative Matrix Factorization)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
- [KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
- [SparsePCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html)

### 🔹 **Feature Selection**
- [SelectKBest, SelectPercentile](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)
- [RFE (Recursive Feature Elimination)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
- [Sequential Feature Selector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)
- [VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)
- [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
- [GenericUnivariateSelect](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html)

---

## 4. **Model (Estimator)**

### 🔹 **Linear Models**
- [LinearRegression (+ Ridge, Lasso, ElasticNet)](https://scikit-learn.org/stable/modules/linear_model.html)
- [LogisticRegression (+ regularized variants)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [SGDClassifier / SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

### 🔹 **Tree-Based Models**
- [DecisionTreeClassifier / DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [RandomForestClassifier / RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [GradientBoostingClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [HistGradientBoostingClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [AdaBoostClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [ExtraTreesClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

### 🔹 **Kernel-Based Models**
- [SVC / SVR (Support Vector Machines)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [GaussianProcessClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)

### 🔹 **Nearest Neighbors**
- [KNeighborsClassifier / KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

### 🔹 **Neural Network-Based Models**
- [MLPClassifier / MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

### 🔹 **Naive Bayes**
- [Naive Bayes models](https://scikit-learn.org/stable/modules/naive_bayes.html)

### 🔹 **Stacking & Ensemble**
- [StackingClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [VotingClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [BaggingClassifier / Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)

---

## 5. **Selection (Final Step, Optional: Model/Feature Selection or Meta-Modeling)**

- 🔹 **Cross-Validation Splitters**
  - [StratifiedKFold, KFold, GroupKFold, etc.](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- 🔹 **Grid/Randomized Search**
  - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
  - [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- 🔹 **Parameter Search with Successive Halving**
  - [HalvingGridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.experimental.enable_halving_search_cv.html)
- 🔹 **CalibratedClassifierCV**
  - [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- 🔹 **Meta-Model Selection**
  - [Stacking, Voting, Bagging (see above)](https://scikit-learn.org/stable/modules/ensemble.html)

---

**For more details and all possible options, check the [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html).**