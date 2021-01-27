# Commodity-News-Corpus

This folder contains the raw files of the Commodity News Corpus, which is made up of a pair of files : a text file (.txt) containing the news article and a annotation file (.ann) containing the annotation details.

## Summary information
- 21 Entity types covering both named and nominal entities
- 19 event types
- 21 Argument roles 
For complete list of the above, please refer to Event Annotation Guidelines.pdf

The diagram below shows the annotation details using the tool called Brat Annotation Tool.

![Annotation](brat_annotation.png)


## Preprocessing
In data pre-processing, the annotation information in Brat standoff format (.ann file) is combined with the text (.txt file) to produce a corresponding .json file as input to the event extraction model.
