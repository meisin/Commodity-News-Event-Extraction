# Commodity News Event Extraction 

## Introduction
This repository contains PyTorch code for the paper entitled **"Effective Use of Pre-trained Language Models and Graph Convolution Network over Contextual Sub-dependency Parse Tree for Event Extraction in Commodity News"**

This paper introduces the use of pre-trained language models, eg: BERT and Graph Convolution Network (GCN) over a sub-dependency parse tree, termed here as **Contextual Sub-Tree** for event extraction in Commodity News. Below is a diagram showing the overall architecture of the proposed solution. 

![Architecture](fig/architecture_without_polaritymodality.png)

The events found in Commodity News are group into three main categories:
1. Geo-political events
2. Macro-economic events
3. Commodity Price Movement events

## Requirements
1. Python 3 (own version is 3.7.4)
2. PyTorch 1.2
3. Transformer (own version is 2.11.0) - [Huggingface](https://huggingface.co/transformers/)

To install the requirements, run ```pip install -r requirements.txt```.

## Repository contents
- ```ComBERT``` folder contains the link to download ComBERT model.
- ```dataset``` folder contains training data- ```event_extraction_train.json``` and testing data- ```event_extraction_test.json```
- ```data``` folder contains (1) ```const.py``` file with Event Labels, Entity Labels, Argument Role Labels and other constants and (2) ```data_loader.py``` with functions relating to the loading of data.
- ```utils``` folder contains helper functions and Tree structure related functions.
- ```model``` folder contains the main Event Extraction Model ```event_extraction.py``` and Graph Convolution Model ```graph_convolution.py```
- ```runs``` folder contains the output of the executions (see Ouput section below for details)

## How to run the codes
Run ```run_train.bat ```

## Output
The results are written to (1) Tensorboard and (2) "runs/logfiles/output_XX.log' where XX is the system date and timestamp. Results include
1. Training loss
2. Evaluation loss
3. Event Trigger classification Accuracy, Precision, Recall and F1 scores.
4. Argument Role classification Accuracy, Precision, Recall and F1 scores.

To access results on Tensorboard, first you need to have Tensorboard install and to bring up to bring up tensorboardX, use this command: ```tensorboard --logdir runs```

## Results
|       Argument role      | precision |  recall  | f1-score |
| NONE                     |   0.95    |   0.97   |   0.96   |
| Attribute                |   0.94    |   0.88   |   0.91   |
| Item                     |   0.96    |   0.88   |   0.92   |
| Final_value              |   0.75    |   0.81   |   0.78   |
| Initial_reference_point  |   0.67    |   0.71   |   0.69   |
| Place                    |   0.86    |   0.81   |   0.83   |
| Reference_point_time     |   0.93    |   0.86   |   0.89   |
| Difference               |   0.87    |   0.85   |   0.86   |   
| Supplier_consumer        |   0.87    |   0.81   |   0.84   |
| Imposer                  |   1.00    |   0.78   |   0.88   |
| Contract_date            |   0.62    |   0.71   |   0.67   |
| Type                     |   0.97    |   0.88   |   0.92   |
| Imposee                  |   0.92    |   1.00   |   0.96   |
| Impacted_countries       |   0.97    |   0.94   |   0.96   |
| Initial_value            |   0.83    |   0.71   |   0.77   |
| Duration                 |   0.92    |   0.96   |   0.94   |
| Situation                |   1.00    |   0.75   |   0.86   |
| Participating_countries  |   1.00    |   1.00   |   1.00   |
| Forecaster               |   0.85    |   1.00   |   0.92   |
| Forecast                 |   0.95    |   0.87   |   0.91   |


| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |

## Citation
If you find the codes or the paper useful, please cite using the following:
