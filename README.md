# Metapath-Fused Heterogeneous Graph Network for Molecular Graph Regression

## Brief Introduction

We proposes to model molecules as heterogeneous graphs and utilize meta-paths to capture important chemical structures in graphs. We construct meta-path based connections and decompose the heterogeneous graph into subgraphs according to relational types. And an hierarchical attention strategy is designed to aggregate heterogeneous information at neighborhood-level and relation-level. Consequently, our method is more natural and suitable to learn the heterogeneous graph structures in molecules. Here is the demo code of our model.

## Environment Requirement

Our code is written in Python3.6.13. The required packages are listed in “requirements.txt”, including

- pytorch == 1.8.1
- pytorch_geometric == 1.7.2
- numpy ==1.19.2
- rdkit == 2021.03.3

## How to run

- Datasets

  The original datasets are offered in "data/QM9" and "data/ZINC".
- First, process the meta-path connectivity construction using the original datasets. Transform the original molecules into heterogeneous graph datasets. And the generated data should be saved as raw files in "data/Dataset_QM9/raw" and "data/Dataset_ZINC/raw" respectively.

Metapath generation on QM9:
```python
python src/qm9_preprocess/QM9_Metapath.py
```

Metapath generation on ZINC:
```python
python src/qm9_preprocess/QM9_Metapath.py
```

```
python src/zinc_preprocess/zinc_Metapath.py
```

- Run the model on the processed datasets:

```python
python src/run_zinc.py
```

```python
python src/run_qm9.py
```
