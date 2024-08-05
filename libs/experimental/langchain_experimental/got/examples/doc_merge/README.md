# Document Merging

The use case in this directory generates new Non-Disclosure Agreement (NDA) based on several input ones that partially overlap in terms of their contents. 
We provide implementations of five different approaches:
- IO
- Chain-of-Thought (CoT)
- Tree of Thought (ToT)
- Graph of Thoughts (GoT):
  - GoT: aggregation of fully merged NDAs
  - GoT2: aggregation of partially merged NDAs

## Data

We provide an input file with 50 samples: `documents.csv`.

## Execution

The file to execute the use case is called
`doc_merge.py`. In the main body, one can
select the specific samples to be run (variable samples) and the
approaches (variable approaches). It is also possible to set a budget in
dollars (variable budget).

The Python scripts will create the directory `result`, if it is not
already present. In the `result` directory, another directory is created
for each run: `{name of LLM}_{list of approaches}_{day}_{start time}`.
Inside each execution specific directory two files (`config.json`,
`log.log`) and a separate directory for each selected approach are
created. `config.json` contains the configuration of the run: input data,
selected approaches, name of the LLM, and the budget. `log.log` contains
the prompts and responses of the LLM as well as additional debug data.
The approach directories contain a separate json file for every sample
and the file contains the Graph Reasoning State (GRS) for that sample.

## Plot Data

Change the results directory in line 158 of `plot.py` and run `python3
plot.py` to plot your data.
