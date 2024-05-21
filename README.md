# ExplGNNswithDL: Explaining Graph Neural Networks (GNN) using Class Expressions (CE) from Description Logics (DL)

We introduce a methodology to explain GNNs using Class Expressions (CE). Our approach is designed to work on heterogeneous datasets with diverse node types, edge types, and node features. The core idea is to leverage Class Expressions (CE) from the description logic EL, which provides support for intersections and exist-relations.

## TL;DR
We explain graph neural networks using class expressions from description logic, which are found by generating graphs from them and averaging the GNN output.

## Overview

Our primary goal is to identify a CE which is valid for graphs that maximize the given GNN's output. The approach comprises the following steps:

1. **Random Class Expression Generation**: Start by creating a random class expression (from the description logic EL). We ensure every individual in the CE has a class.
2. **Graph Creation**: Develop multiple graphs that satisfy the generated CE. In the first step, we include an edge for each relation present in the CE.
3. **GNN Evaluation**: Test the GNN on these graphs and use the GNN's output as feedback to refine the CE.
4. **CE Optimization**: We plan to implement a simple optimization algorithm, to find the CE which gives rise to the best explanatory graph. We use this CE to explain the GNN for the given dataset.



## Abstract

Graph Neural Networks (GNNs) have emerged as
powerful tools for node and graph classification on
graph-structured data, but they lack explainability,
particularly when it comes to understanding their
behavior at global level. Current research mainly
utilizes subgraphs of the input as local explanations and generates graphs as global explanations.
However, these graph-based methods have limited
expressiveness in explaining a class with multiple
sufficient explanations. To achieve more expressive explanations, we propose utilizing class ex-
pressions (CEs) from the field of description logic
(DL). We demonstrate our approach on heterogeneous graphs with different node types, which we
explain with CEs in the description logic EL. To
identify the most fitting explanation, we construct
multiple graphs for each CE and aggregate their
GNN output. In contrast to optimizing the GNN, as
a baseline we optimize the CE’s fidelity directly on
a validation dataset. Our approach offers a more expressive mechanism for explaining GNNs in node
classification tasks and addresses the limitations of
existing graph-based explanations

## Explanations to the Code

#### Installation
Run the file `requirements.txt` for installation with pip. Then run the shell `./run_egel.sh` or directly `main.py`

#### Overview of the results
These can be found in the following folders:

`content/plots` contains all plots from Beam search using graph generation

`GroundTruth...` folders contain all plots to the ground truth

`Score_FidelityHeteroBAShapes` contains all plots tfrom Beam search using fidelity optimization

`results_txt` contains all log files, from the GNN on the best CEs with manipulated edges (deleted all edges of a certain type, ... )


#### Overview of the Files
There is a vast summary of files, but some are very important:

`create_random_ce.py` describes all functions, which are needed to create and mutate one CE, with some additional utility functions.

`generate_graphs.py` describes all functions which are needed to create heterogeneous PyTorch-geometric graphs from a CE.

`main.py` brings all functions together and describes the Beam-Search.

`evaluation` describes the scoring function for beam search, as well as accuracy and fidelity.

`datasets.py` describes the generation of the dataset heteroBAShapes.

`visualization.py` describes all functions needed to visualize the results.

#### Functions from the paper

1. **Mutate CE** refers to the function `mutate_ce` in `create_random_ce.py`
2. **Beam Search** refers to the function `beam_search` in `main.py`
3. **Create Graph** refers to the function `get_graph_from_ce` in `create_random_ce`
4. **Accuracy** refers to the function `get_accuracy_baheteroshapes` in `evaluation.py`

#### Parameters to set in `main.py`
Some parameters can be set, to just run parts of the code:

`retrain_GNN_and_dataset`: IF the GNN should be retrained and the dataset should be re-created. Otherwise it is taken from previous runs from the hard drive.

`run_beam_search`: If beam search should be run, using GNN maximization as scoring function

`run_tests_ce`: If the CEs should be evaluated on manipulated scores

`run_beam_search_fidelity`: If beam search maximizing fidelity should be run.


Further parameters contain the setting:

`number_of_ces` gives the bandwidth of beam search

`number_of_graphs` gives the number of graphs which are created for one CE to evaluate the GNN ontop.

`lambdaone` gives the regularization parameter for the length of the CE

`aggr_fct` gives the aggregation function used to score the GNN on the graphs (mean and max implemented).

`size_dataset` gives the size of the dataset; 10% of the size will be the number of motifs added to the dataset to create a HeteroBAShapes dataset.

`num_top_results` gives the number of results, which should be visualized in the end.

