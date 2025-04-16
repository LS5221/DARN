# Title: Federated Incomplete Tabular Data Prediction with Missing Complementarity
## Introduction

This folder holds the source code of the proposed system DARN , including

  - Model implementations (in ./models)
  - Utility functions (in ./utils)
  - Main algorithm (in ./train_caFEDAVG.py)
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset Description

All of the original datasets can refer to the links:

  - News: https://archive.ics.uci.edu/dataset/332/online+news+popularity          
  - HI: https://www.openml.org/search?type=data&status=active&id=23512&sort=runs       
  - gas: https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures   
  - temperature: https://archive.ics.uci.edu/dataset/514/bias+correction+of+numerical+prediction+model+temperature+forecast 


## Usage

  - You can use scripts in run.sh to run all the experiments