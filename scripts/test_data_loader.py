import os
import sys
from pathlib import Path
from data_loader.csqa_dataset import CSQADataset

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# load data
dataset = CSQADataset('/data/final/csqa/sample')
vocabs = dataset.get_vocabs()
val_data = dataset.get_data()