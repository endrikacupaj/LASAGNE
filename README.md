# Conversational Question Answering over Knowledge Graphs with Transformer and Graph Attention Networks

This paper addresses the task of (complex) conversational question answering over a knowledge graph. For this task, we propose LASAGNE (muLti-task semAntic parSing with trAnsformer and Graph atteNtion nEtworks). It is the first approach, which employs a transformer architecture extended with Graph Attention Networks for multi-task neural semantic parsing. LASAGNE uses a transformer model for generating the base logical forms, while the Graph Attention model is used to exploit correlations between (entity) types and predicates to produce node representations. LASAGNE also includes a novel entity recognition module which detects, links, and ranks all relevant entities in the question context. We evaluate LASAGNE on a standard dataset for complex sequential question answering, on which it outperforms existing baseline averages on all question types.  Specifically, we show that LASAGNE improves the F1-score on eight out of ten question types; in some cases, the increase in F1-score is more than 20% compared to the state of the art.

![LASAGNE](image/lasagne_architecture.png?raw=true "LASAGNE architecture")

LASAGNE (Multi-task Semantic Parsing with Transformer and Graph Attention Networks) architecture. It consists of three modules: 1) A semantic parsing-based transformer model, containing a contextual encoder and a grammar guided decoder using the proposed grammar. 2) An entity recognition module, which identifies all the entities in the context, together with their types, linking them to the knowledge graph. It filters them based on the context and permutes them, in case of more than one required entity. Finally, 3) a graph attention-based module that uses a GAT network initialised with BERT embeddings to incorporate and exploit correlations between (entity) types and predicates. The resulting node embeddings, together with the context hidden state and decoder hidden state, are used to score the nodes and predict the corresponding type and predicate.

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
### Download
We evaluate LASAGNE on [CSQA](https://amritasaha1812.github.io/CSQA/) dataset. You can download the dataset from [here](https://amritasaha1812.github.io/CSQA/download/).

### Wikidata Knowlegde Graph
CSQA dataset is based on Wikidata [Knowlegde Graph](https://www.wikidata.org/wiki/Wikidata:Main_Page), the authors provide a preproccesed version of it which can be used when working with the dataset.
You can download the preprocessed Wikidata knowlegde graph files from [here](https://zenodo.org/record/4052427#.YBU7xHdKjfZ).
After dowloading you will need to move them under the [knowledge_graph](knowledge_graph) directory.

We prefer to merge some JSON files from the preprocessed Wikidata, for accelerating the process of reading all the knowledge graph files. In particular, we create three new JSON files using the script [prepare_data.py](scripts/prepare_data.py). Please execute the script as below.
``` bash
# prepare knowlegde graph files
python scripts/prepare_data.py
```

### Inverted index on Wikidata entities
For building an inverted index on wikidata entities we use [elastic](https://www.elastic.co/) search. Consider the script file [csqa_elasticse.py](scripts/csqa_elasticse.py) for doing so.

### Annotate Dataset
Next, using the preproccesed Wikidata files we can annotate CSQA dataset with our grammar. At the same time we also annotate the entity spans for all utterances.
``` bash
# annotate CSQA dataset with entity spans and our grammar
python annotate_csqa/preprocess.py --partition train --annotation_task all --read_folder /path/to/CSQA --write_folder /path/to/write
```

## BERT embeddings
Before training the framework, we need to create BERT embeddings for the knowledge graph (entity) types and relations. You can do that by running.
``` bash
# create bert embeddings
python scripts/bert_embeddings.py
```

## Train Framework
For training you will need to adjust the paths in [args](args.py) file. At the same file you can also modify and experiment with different model settings.
``` bash
# train framework
python train.py
```

## Test
For testing we have two steps.
### Generate Actions
First, we generate the actions and save then in JSON file using the trained model.
``` bash
# generate actions for a specific question type
python inference.py --question_type Clarification
```

### Execute Actions
Second, we execute the actions and get the results from Wikidata files.
``` bash
# execute actions for a specific question type
python action_executor/run.py --file_path /path/to/actions.json --question_type Clarification
```

## License
The repository is under [MIT License](LICENCE).

## Cite
```bash
@inproceedings{
    kacupaj2021lasagne,
    title={Conversational Question Answering over Knowledge Graphs with Transformer and Graph Attention Networks},
    author={Endri Kacupaj and Joan Plepi and Kuldeep Singh and Harsh Thakkar and Jens Lehmann and Maria Maleshkova},
    booktitle={16th conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)},
    year={2021}
}
```