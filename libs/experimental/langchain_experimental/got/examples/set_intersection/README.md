# Set Intersection

The use case in this directory computes the intersection of two input
sets. We provide implementations of five different approaches for 32, 64
and 128 elements:
- IO
- Chain-of-Thought (CoT)
- Tree of Thought (ToT):
  - ToT: wider tree, meaning more branches per level
  - ToT2: tree with more levels, but fewer branches per level
- Graph of Thoughts (GoT)

## Data

We provide input files with 100 precomputed samples for each set length:
`set_intersection_<number of elements>.csv`. It is also possible to use
the data generator `dataset_gen_intersection.py` to generate additional or
different samples. The parameters can be updated in lines 24 to 28 of
the main body:
- set_size = 32 # size of the generated sets
- int_value_ubound = 64 # (exclusive) upper limit of generated numbers
- seed = 42 # seed of the random number generator
- num_sample = 100 # number of samples
- filename = 'set_intersection_032.csv' # output filename

## Execution

The files to execute the use case are called
`set_intersection_<number of elements>.py`. In the main body, one can
select the specific samples to be run (variable sample) and the
approaches (variable approaches). It is also possible to set a budget in
dollars (variable budget).
The input filename for the samples is currently hardcoded to
`set_intersection_<number of elements>.csv`, but can be updated in the
function `run`.

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

Change the results directory in line 170 of `plot.py` and update the
length parameter in the subsequent line and run `python3 plot.py` to
plot your data.
