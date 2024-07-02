# Keyword Counting

The use case in this directory computes the frequencies of occurring countries 
in a long passage of text. We provide implementations of seven different approaches:
- IO
- Chain-of-Thought (CoT)
- Tree of Thought (ToT):
  - ToT: wider tree, meaning more branches per level
  - ToT2: tree with more levels, but fewer branches per level
- Graph of Thoughts (GoT):
  - GoT4: split passage into 4 sub-passages
  - GoT8: split passage into 8 sub-passages
  - GoTx: split by sentences

## Data

We provide an input file with 100 samples: `countries.csv`. It is also possible to use
the data generator `dataset_gen_countries.py` to generate additional or
different samples (using GPT-4). The parameters can be updated on line 54 (number of samples to be generated). 
Note that not every generated sample will be included in the dataset, as each sample is 
additionally tested for validity (observe script output for details).

## Execution

The file to execute the use case is called
`keyword_counting.py`. In the main body, one can
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

Change the results directory in line 150 of `plot.py` and run `python3
plot.py` to plot your data.
