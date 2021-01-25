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
4. Stanford CoreNLP

To install the requirements, run    pip -r requirements.txt.

## How to run the codes
