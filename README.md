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

Graph Neural Networks (GNNs) have emerged as powerful tools for node and graph classification on graph-structured data, but they lack explainability, particularly when it comes to understanding their behavior at global level. Current research mainly utilizes subgraphs of the input as local explanations and generates graphs as global explanations. However, these graph-based methods have limited expressiveness in explaining a class with multiple sufficient explanations. To achieve more expressive explanations, we propose utilizing class expressions (CEs) from the field of description logic (DL). We demonstrate our approach on heterogeneous graphs with different node types, which we explain with CEs in the description logic EL. To identify the most fitting explanation, we construct multiple graphs for each CE and calculate the average GNN output among those graphs. Our approach offers a more expressive mechanism for explaining GNNs in node classification tasks and addresses the limitations of existing graph-based explanations.

## Explanations to the Code

#### Installation
Install `requirements.txt` in some Python version (working: 3.11, linux, macOS). 

Troubleshoot: 
- Delete all  imports of `dgl` -> This library is not working in this environment
- Install missing packages, if asked so.

#### Overview of the Files
There is a vast summary of files, but only some are important:

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


#### How to cite

``` @xgnn-dl,
  author       = {Dominik K{\"{o}}hler and
                  Stefan Heindorf},
  title        = {Utilizing Description Logics for Global Explanations of Heterogeneous
                  Graph Neural Networks},
  journal      = {CoRR},
  volume       = {abs/2405.12654},
  year         = {2024}
}
```
