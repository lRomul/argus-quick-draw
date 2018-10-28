import pandas as pd
import tqdm
import json
import random
import numpy as np
from os.path import join

from src import config

SAVE_NAME = 'val_key_ids_001'
VAL_SPLIT = 0.001
SAVE_PATH = join(config.DATA_DIR, SAVE_NAME+'.json')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    val_key_ids = []
    for cls in tqdm.tqdm(config.CLASSES):
        class_df = pd.read_csv(config.CLASS_TO_CSV_PATH[cls])
        cls_size = round(class_df.shape[0] * VAL_SPLIT)
        val_cls_key_ids = np.random.choice(class_df.key_id, size=cls_size, replace=False)
        val_key_ids += val_cls_key_ids.tolist()

    with open(SAVE_PATH, 'w') as file:
        file.write(json.dumps(val_key_ids))

    print("Validation key ids saved to", SAVE_PATH)
