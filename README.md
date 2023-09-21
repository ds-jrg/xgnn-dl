# ExplGNNswithDL: Explaining Graph Neural Networks (GNN) using Class Expressions (CE) from Description Logics (DL)

We introduce a methodology to explain GNNs using Class Expressions (CE). Our approach is designed to work on heterogeneous datasets with diverse node types, edge types, and node features. The core idea is to leverage Class Expressions (CE) from the description logic EL, which provides support for intersections and exist-relations.

## Overview

Our primary goal is to identify a CE that is valid for graphs that maximize the given GNN's output. The approach comprises the following steps:

1. **Random Class Expression Generation**: Start by creating a random class expression (from the description logic EL). We ensure, that every individual in the CE has a class.
2. **Graph Creation**: Develop multiple graphs that satisfy the generated CE. In the first step, we include an edge for each relation present in the CE.
3. **GNN Evaluation**: Test the GNN on these graphs and use the GNN's output as feedback to refine the CE.
4. **CE Optimization**: We plan to implement a simple optimization algorithm, to find the CE which gives rise to the best explanatory graph. We use this CE to explain the GNN for the given dataset.
