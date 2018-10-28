import os
from os.path import join

DATA_DIR = '/workdir/data/'
TRAIN_SIMPLIFIED = join(DATA_DIR, 'train_simplified')
TEST_SIMPLIFIED_PATH = join(DATA_DIR, 'test_simplified.csv')
BASE_SIZE_SIMPLIFIED = 256
SAMPLE_SUBMISSION = join(DATA_DIR, 'sample_submission.csv')

CLASSES = sorted([p[:-4] for p in os.listdir(TRAIN_SIMPLIFIED) if p.endswith('.csv')])
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
CLASS_TO_CSV_PATH = {cls: join(TRAIN_SIMPLIFIED, cls+'.csv') for cls in CLASSES}
