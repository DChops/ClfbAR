# ClfbAR

**Classification By Association Rules Mining (CARS) Algorithm**

Algorithm adapted from *Liu et al. 1998*

### Features of our CARS Classifier: 

* High explainability in the form of Decision Rules
* Achieves comparable perfromance to more advanced classifiers on small datasets (See Benchmarks)
* Intrinsic Null handling capabilities (no need for imputations or dropping)
* Handles numeric data using binning functionalities

## Usage:

**Using CARS Classifier**
```python
from clfbar.clfbar import CarClassifier

### Set MinSup and MinConf or use defaults
c = CarClassifier(0.8,0.6)

### Fit the classifier
c.fit(X_train,y_train)

### Display Association rules learnt from the training data
c.rules

### Predict on test data
c.predict(X_val)
```

**Using Binning Helpers:**
Transform Pandas Series from numeric dtypes into categoriecal dtype


1. Use Fisher-Jenks Binning Algorithm
```python
from clfbar.binners import jenks_binner

### Set MAX_CLASSES & THRESHOLD or use default values
jnb = jenks_binner(5, 0.8)

### Fit and Transform numeric data into categorries
transformed_ser = jnb.fit(ser)

### Use binning from training without fitting on new data
transformed_test_ser = jnb.transform(test_ser)
```

2. Use Equal Frequency Binning
```python
transformed_data = equal_freq_binner(data_frame, bins=5)
```

### Instalation: 
```python
git clone https://github.com/DChops/ClfbAR.git
```

### References
* Liu, B., Hsu, W., & Ma, Y. (1998). Integrating Classification and Association Rule Mining. KDD.
* jenkspy library: https://pypi.org/project/jenkspy/