import argparse
import os
import json
import glob
from pathlib import Path
import re
import json
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score
import pandas as pd
import numpy as np
from scipy.signal import medfilt

parser = argparse.ArgumentParser("scoring")
parser.add_argument("preds_dir")
parser.add_argument("diarization_file")
# python score_ap.py /sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/exp/tcn/train_logs /sdb/home/lishaojie/lsj_code/OSDC-master/egs/AMI/data/diarization_eval/
# python score_ap.py /home/lsj/OSDC-master/egs/exp/tcn/preds_mfcc_0.5wlossLR20_dev /home/lsj/OSDC-master/egs/data/diarization_dev/dev.json

# python score_ap.py /home/lsj/OSDC-Loss/egs/exp/tcn/preds_tcn_csdopp_dev /home/lsj/OSDC-SACC/egs/data/diarization_dev/dev.json

# python score_ap.py /home/lsj/OSDC-master/egs/exp/tcn/preds_micro_csd8_topmax1_ce_dev /home/lsj/OSDC-master/egs/data/diarization_dev/dev.json
# python score_ap.py /home/lsj/OSDC-master/egs/exp/tcn/preds_mfcc_0.5wlossLR20_0.15mseloss_dev /home/lsj/OSDC-master/egs/data/diarization_dev/dev.json
# python score_ap.py /home/lsj/OSDC-master/egs/exp/tcn/preds_micro_csd8_topmax10_0.5wlossLR20_0.15mse_dev /home/lsj/OSDC-master/egs/data/diarization_dev/dev.json
def build_target_vector(sess_diarization, subsample=160):

    # get maxlen
    maxlen = max([sess_diarization[spk][-1][-1] for spk in sess_diarization.keys()])
    dummy = np.zeros(maxlen//subsample, dtype="uint8")

    for spk in diarization[sess].keys():
        if spk == "garbage":
            continue
        for s, e in diarization[sess][spk]:
            s = int(s/subsample)
            e = int(np.ceil(e/subsample))
            dummy[s:e] += 1
    return dummy


def one_hot(a, num_classes):

    a = np.clip(a, 0, 2)
    # a = np.clip(a, 0, 4)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def score(preds, target_vector):
    # target_vector[-1] = 4
    target_vector[-1] = 2
    target_vector_cat = target_vector
    # print(f'target_vector_cat:{target_vector_cat}')
    target_vector = one_hot(target_vector, preds.shape[0])
    
    # print(f'target_vector:{target_vector.shape}')
    minlen = min(target_vector.shape[0], preds.shape[-1])
    target_vector = target_vector[:minlen, :]
    target_vector_cat = target_vector_cat[:minlen]
    preds = preds[:, :minlen].T
    print(f'preds:{preds.shape}')

    # count_ap = average_precision_score(target_vector, preds, average=None)
    osd_ap = average_precision_score(target_vector_cat >= 2, np.sum(preds[:, 2:], -1))
    vad_ap = average_precision_score(target_vector_cat >= 1, np.sum(preds[:, 1:], -1))


    # return count_ap, vad_ap, osd_ap
    return  vad_ap, osd_ap

def process_preds(preds):

    # average all from same session
    # apply medfilter
    mat = []
    minlen = np.inf
    for i in preds:
        tmp = np.load(i)[0]
        minlen = min(minlen, tmp.shape[-1])
        mat.append(tmp)

    mat = [x[:minlen] for x in mat]
    mat = np.mean(np.stack(mat), 0)

    return mat #medfilt(mat, 5)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.diarization_file, "r") as f:
        diarization = json.load(f)

    preds_hash = {}
    preds = glob.glob(os.path.join(args.preds_dir, "*.npy"))

    for p in preds:
        session = Path(p).stem.split(".")[0]
        if session not in preds_hash.keys():
            preds_hash[session] = [p]
        else:
            preds_hash[session].append(p)

    scores = {}
    for sess in diarization.keys():
        if sess not in preds_hash.keys():
            continue
        print(f'sess:{sess}')

        target_vector = build_target_vector(diarization[sess])
        print(f'target_vector.shape:{target_vector.shape}')
        print(f'target_vector:{target_vector[:50]}')
        preds = preds_hash[sess]
        preds = process_preds(preds)
        # count_ap, vad_ap, osd_ap = score(preds, target_vector)
        vad_ap, osd_ap = score(preds, target_vector)
        # scores[sess] = {"count_ap": count_ap, "vad_ap": vad_ap, "osd_ap": osd_ap }
        scores[sess] = {"vad_ap": vad_ap, "osd_ap": osd_ap }

    dt = pd.DataFrame.from_dict(scores)
    # scores["TOTAL"] = {"count_ap": dt.iloc[0, :].mean(), "vad_ap": dt.iloc[1, :].mean(), "osd_ap": dt.iloc[2, :].mean()}
    scores["TOTAL"] = {"vad_ap": dt.iloc[0, :].mean(), "osd_ap": dt.iloc[1, :].mean()}
    print(f'scores["TOTAL"]:{scores["TOTAL"]}')
    dt = pd.DataFrame.from_dict(scores).to_json(os.path.join(args.preds_dir, "APs.json"))














