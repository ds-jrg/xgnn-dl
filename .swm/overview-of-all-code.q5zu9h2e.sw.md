---
title: Overview of all Code
---
<SwmPath>[main.py](/main.py)</SwmPath> : Here we run all experiments

<SwmPath>[main_utils.py](/main_utils.py)</SwmPath> : all functions needed to run experiments.&nbsp;

&nbsp;

&nbsp;

How is a new dataset created?

<SwmPath>[datasets.py](/datasets.py)</SwmPath> : All utility functions to create a synthetic dataset.

&nbsp;

Creating CEs:

<SwmPath>[create_random_ce.py](/create_random_ce.py)</SwmPath> : All utility functions to randomly create a new Class Expression (CE). The Idea is to start with a pre-selected class and then mutate this class using methods from <SwmToken path="/create_random_ce.py" pos="402:2:2" line-data="class Mutation:">`Mutation`</SwmToken> to enlarge it.

- <SwmToken path="/create_random_ce.py" pos="26:2:2" line-data="class CEUtils():">`CEUtils`</SwmToken> is the class summarizing all Utility functions for CEs (all static methods)
- <SwmToken path="/create_random_ce.py" pos="402:2:2" line-data="class Mutation:">`Mutation`</SwmToken> is the class to handle all Mutations, initialized by ways to mutate:
  - possible classes (node types)
  - possible object properties (edge types)
  - maximal depth of outcoming CE

&nbsp;

<SwmToken path="/beamsearch.py" pos="51:2:2" line-data="class BeamSearch():">`BeamSearch`</SwmToken>: To search the best CEs to describe a dataset or gnn, we use beam search:

- It is initialized by the beam search parameters, the GNN to explain, the dataset, the scoring function and the maximal depth of the CEs. It outputs the <SwmToken path="/beamsearch.py" pos="80:3:3" line-data="        self.beam_width = beam_width">`beam_width`</SwmToken> best CEs, giving the <SwmToken path="/beamsearch.py" pos="83:3:3" line-data="        self.scoring_function = scoring_function">`scoring_function`</SwmToken>.

&nbsp;

## Creating and importing datasets

- <SwmPath>[syntheticdatasets.py](/syntheticdatasets.py)</SwmPath> is the file to create all kind of synthetic datasets in the class <SwmToken path="/syntheticdatasets.py" pos="5:2:2" line-data="class SyntheticDatasets():">`SyntheticDatasets`</SwmToken>. Currently:
  - A house motif <SwmToken path="/syntheticdatasets.py" pos="10:3:3" line-data="    def new_dataset_house(num_nodes, num_motifs=None, num_edges=3):">`new_dataset_house`</SwmToken> , with node types A,B,C (top to bottom) and D (for nodes not in the house). The dataset takes a BA Graph and adds house motifs to it. Then it creates node types, according to the position in the graph, and randomly changes node types in the BA Graph. The goal for the GNN is to predict for a certain node type, whether it is in the BAGraph or the motif.
- <SwmPath>[datasets.py](/datasets.py)</SwmPath> Has all utility functions to make datasets, divide them into test, validation, and training data, and convert them from one standard to the other (nxgraph <-> PyG <-> PyG Heterodata).

## 

All GNNs are trained in the class <SwmToken path="/models.py" pos="36:2:2" line-data="class GNN_datasets():">`GNN_datasets`</SwmToken> from the file <SwmPath>[models.py](/models.py)</SwmPath>.&nbsp;

&nbsp;

&nbsp;

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBeGdubi1kbCUzQSUzQWRzLWpyZw==" repo-name="xgnn-dl"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>
