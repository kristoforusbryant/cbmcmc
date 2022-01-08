# cbmcmc
MCMC Algorithm for Gaussian Graphical Models using Cycle Bases

To replicate the experiment results presented in the manuscript, first, in the current directory run
```
export PYTHONPATH=:$(pwd):$PYTHONPATH
cd breastcancer/
```

Then, the dataset, edge and star cycle bases MCMC estimates, and plots can be obtained by executing the scripts 
named `get_data.r`, `mcmc.py` and `plot.py`, respectively.
```
Rscript get_data.r 
python mcmc.py
python plot.py
```

The script `plot_graph.py` plots percentile graphs obtained from the edge and star cycle bases algorithms. 
PyGraphviz is required for this script.

