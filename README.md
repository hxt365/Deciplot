# Deciplot
This is a handy tool that helps you better visualize your data by plotting 2D decision boundaries, using any sklearn classifier models.

## Installation
Run the following to install:
```python
pip install deciplot
```

## Usage
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from deciplot import DeciPlot2D

# Get iris dataset from sklearn
iris = load_iris()
X, y, feature_names, target_names = iris['data'], iris['target'], iris['feature_names'], iris['target_names']

# Initialize a classifier model for plotting decision boundaries
dtree = DecisionTreeClassifier()

# Initialize an DecPlot2D object
dp2d = DeciPlot2D(X, y, feature_names, target_names, dtree)

# Plot and visualize decision boundaries
dp2d.plot(figsize=(10, 10))
```
After running this code, we'll get this beautiful plot:
<p align="center">
<img src='https://user-images.githubusercontent.com/16115992/97483366-62cfa280-1968-11eb-89e4-b112f619765e.png' alt='Decision boundary plot' height=400px/>
</p>
