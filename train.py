from dataloader import *


icat = InCategoryClutterDataset('train', 227, "/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set")
i,s,o,c = icat[0]