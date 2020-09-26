# Conversational Question Answering over Knowledge Graphs with Transformer and Graph Attention Networks

## Requirements and Setup

Python version >= 3.7

PyTorch version >= 1.6.0

PyTorch Geometric (PyG) >= 1.6.1

``` bash
# clone the repository
git clone https://github.com/endrikacupaj/LASAGNE.git
cd LASAGNE
pip install -r requirements.txt
```

## CSQA dataset
Our framework was evaluated on [CSQA](https://amritasaha1812.github.io/CSQA/) dataset. You can download the dataset from [here](https://amritasaha1812.github.io/CSQA/download/).

## Wikidata Knowlegde Graph
Since CSQA is based on Wikidata [Knowlegde Graph](https://www.wikidata.org/wiki/Wikidata:Main_Page), the authors provide a preproccesed version of it which can be used when working with the dataset.
You can download the preprocessed files from here.
After dowloading you will need to move them under the [knowledge_graph](knowledge_graph) directory.

## Annotate Dataset
Next, using the preproccesed Wikidata files we can annotate annotate CSQA dataset with the grammar proposed in our paper. At the same time we also annotate the entity spans for all utterances.
``` bash
# annotate CSQA dataset with proposed grammar and entity spans
python annotate_csqa/preprocess.py --partition train --annotation_task all --read_folder /path/to/CSQA --write_folder /path/to/write
```

## Train Framework
For training you will need to adjust the paths in [args](args.py) file. At the same file you can also modify and experiment with different model settings.
``` bash
# train framework
python train.py
```

## Generate Actions
After model has finished training we perform the inference in 2 steps.
First, we generate the actions and save then in JSON file using the trained model.
``` bash
# generate actions for a specific question type
python inference.py --question_type Clarification
```

## Execute Actions
Second, we execute the actions and get the results from Wikidata files.
``` bash
# execute actions for a specific question type
python action_executor/run.py --file_path /path/to/actions.json --question_type Clarification
```

## License
The repository is under MIT License.

## Cite
Coming Soon!