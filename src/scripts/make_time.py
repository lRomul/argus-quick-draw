import os
import pandas as pd
import multiprocessing as mp

simp_path = '/workdir/data/train_simplified'
raw_path = '/workdir/data/train_raw'
out_dir = '/workdir/data/train_time'

N_WORKERS = mp.cpu_count()

def make_stroke_time(cls_name):
    simp = pd.read_csv(os.path.join(simp_path, cls_name))
    raw = pd.read_csv(os.path.join(raw_path, cls_name))
    for i in range(len(simp)):
        drawing = eval(simp.iloc[i].drawing)
        drawing_raw = eval(raw.iloc[i].drawing)
        tot_time = max(drawing_raw[-1][2][-1], 1)
        for j in range(len(drawing)):
            start_time = drawing_raw[j][2][0] / tot_time
            finish_time = drawing_raw[j][2][-1] / tot_time
            drawing[j] = drawing[j] + [[start_time, finish_time]]
        simp.at[i, 'drawing'] = drawing
    simp.to_csv(os.path.join(out_dir, cls_name))

c = 0
if __name__ == '__main__':
    names = os.listdir(simp_path)
    with mp.Pool(N_WORKERS) as pool:
        for _ in pool.imap_unordered(make_stroke_time, names):
            print(c, "/", len(names))
            c+=1
