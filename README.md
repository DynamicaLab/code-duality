# code-duality
Code for the paper *Duality between predictability and recontructability in complex systems*

[![DOI](https://zenodo.org/badge/371061340.svg)](https://zenodo.org/badge/latestdoi/371061340)

## Requirements and dependencies

* `numpy>=1.20.3`
* `scipy>=1.7.1`
* `psutil>=5.8.0`
* `tqdm>=4.56.0`
* `networkx>=2.8`
* `netrd`
* `seaborn`
* `graphinf>=0.2.4`
* `basegraph==1.0.0`


## Installation
First, clone this repository.
```bash
git clone https://github.com/DynamicaLab/code-duality
```
Second, use pip to install the module.
```bash
pip install ./code-duality
```

## Usage
The package is basically wrapper around `graphinf` which does the heavy lifting for all bayesian computations. The main objects of the package are factories, for instanciating the graph and data models, configs and metrics. The configs should be subclasses of [`Config`](code_duality/config.py), and can also be instanciated using `json` files. Here is an example of config instanciation:

```python

from code_duality import Config

config = Config(x=1, y="Hello world") # A basic config
nested = Config(c=config, name="nested_config") # A nested config, that has a config as an attribute
sequenced = Config(seq1=[1,2,3], seq2=["a", "b"] n=nested) # This is a sequenced config, that can be iterated over. seq1 and seq2 are sequenced parameters
# We take the cartesian product between all sequenced parameters
assert len(sequenced) == 6
for s in sequenced.to_sequence(): 
    print(s) # `s` is a non-sequenced config where the parameter `seq1` and `seq2` are iterated over [1,2,3] and ["a", "b"], respectively.

```

The [`Metrics`](code_duality/metrics/metrics.py`) class is a utility class for performing computations in parallel. All subclasses of [`Metrics`](code_duality/metrics/metrics.py`) come with a `compute` method that performs the computation for a sequenced config. The `compute` must be provided with `n_workers` and `n_async_jobs`, corresponding to the number of available processors and the number of config computations that are allowed to perform in parallel. The subclass [`ExpectationMetrics`](code_duality/metrics/metrics.py) does sampling in parallel and outputs an Monte Carlo expectation. The results are saved in the attribute `data` in the form of a `pandas.DataFrame`.

```python
import pandas as pd, os
from code_duality import Config
from code_duality.metrics import BayesianInformationMetrics, CheckPoint


config = Config.load("path/to/the/config.json")

metrics = BayesianInformationMetrics()
metrics.compute(
    config, 
    n_workers=4, 
    n_async_jobs=1, 
    callbacks=[
        CheckPoint.to_setup(patience=1, savepath="path/to/data/folder") # This checkpoint saves the metrics dataframe in a folder at each iteration
    ])
print(metrics.data) # Prints the metrics dataframe
df = pd.read_pickle(os.path.join("path/to/data/folder"), metrics.shortname + ".pkl") # The metrics saves its measure in a file named after its short name class attribute.
print(df) # Prints the same metircs dataframe as above
```

## Results reproduceability
The results of the paper can all be reproduced using the [`main.py`](scripts/main.py) script. The configs for each figure is located [here](scripts/configs). To run the script, use the command-line interface:

```bash
python main.py --config path/to/the/config.json --n-workers 4 --n-async-joibs 1 --ouput-path path/to/data/folder --save-patience 1
```