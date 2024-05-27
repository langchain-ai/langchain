# Sorting

The use case in this directory sorts the provided list of 
numbers containing numbers from 0 to 9 (duplicates allowed). 
We provide implementations of five different approaches for 
32, 64 and 128 elements:
- IO
- Chain-of-Thought (CoT)
- Tree of Thought (ToT):
  - ToT: wider tree, meaning more branches per level
  - ToT2: tree with more levels, but fewer branches per level
- Graph of Thoughts (GoT):
  - GoT: split into subarrays / sort / merge

## Data

We provide input files with 100 precomputed samples for each list
length: `sorting_<number of elements>.csv`.

## Execution

The files to execute the use case are called
`sorting_<number of elements>.py`. In the main body, one can select the
specific samples to be run (variable sample) and the approaches
(variable approaches). It is also possible to set a budget in dollars
(variable budget).
The input filename for the samples is currently hardcoded to
`sorting_<number of elements>.csv`, but can be updated in the function
`run`.

The Python scripts will create the directory `result`, if it is not
already present. In the 'result' directory, another directory is created
for each run: `{name of LLM}_{list of approaches}_{day}_{start time}`.
Inside each execution specific directory two files (`config.json`,
`log.log`) and a separate directory for each selected approach are
created. `config.json` contains the configuration of the run: input data,
selected approaches, name of the LLM, and the budget. `log.log` contains
the prompts and responses of the LLM as well as additional debug data.
The approach directories contain a separate json file for every sample
and the file contains the Graph Reasoning State (GRS) for that sample.

## Plot Data

Change the results directory in line 171 of `plot.py` and update the
length parameter in the subsequent line and run `python3 plot.py` to
plot your data.
