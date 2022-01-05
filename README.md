# cbmcmc
MCMC Algorithm for Gaussian Graphical Models using Cycle Bases

To replicate the experiment results presented in the manuscript, first run
`export PYTHONPATH=:$(pwd):$PYTHONPATH`
in the current directory and change directory to `breastcancer/`.


Then, the dataset, edge and star cycle bases MCMC estimates, and plots can be obtained by executing the scripts 
named `get_data.r`, `mcmc.py` and `plot.py`, respectively.

The script `plot_graph.py` plots percentile graphs obtained from the edge and star cycle bases algorithms. 
PyGraphviz is required for this script.

