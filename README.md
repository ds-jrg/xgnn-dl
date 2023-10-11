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
