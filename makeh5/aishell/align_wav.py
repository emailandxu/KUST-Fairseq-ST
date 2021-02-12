import os
from functools import partial
from pandas import DataFrame 
import pandas as pd

def make_wav_list(corpus_root=None):
    if corpus_root:
        os.system(f'find {corpus_root} -name "*.wav" > wav.list')
    else:
        os.system('find $PWD -name "*.wav" > wav.list')

def make_wav_trans_excel(subset="train", transcript_path="/mnt/d/corpus/data_aishell/transcript/aishell_transcript_v0.8.txt"):
    make_wav_dict = lambda subset: {os.path.basename(line.strip().replace(".wav","")):line.strip() for line in open("./wav.list",'r') if subset in line}
    make_subset_wav_dict = partial(make_wav_dict, subset=subset)
    make_wav_trans_dict = lambda : dict(line.strip().split(" ", 1) for line in open(f"{transcript_path}","r", encoding="utf-8"))

    wav_trans_dict = make_wav_trans_dict()
    wav_dict = make_subset_wav_dict()

    trans = DataFrame(wav_trans_dict.items())
    wavs = DataFrame(wav_dict.items())

    print(trans)
    print(wavs)

    pd.merge(wavs,trans,on=0).to_excel(subset+".xlsx")

def make_wavs_trans_txt(subset="train"):
    wav_trans= load_wav_trans(f"{subset}")

    f = open(f"./{subset}.wavs",'wt')
    wavs = wav_trans['1_x']
    print("\n".join(wavs), file=f)

    f = open(f"./{subset}.trans",'wt')
    trans = wav_trans['1_y']
    trans = [tran.replace(" ","").strip() for tran in trans]
    print("\n".join(trans), file=f)

def load_wav_trans(subset="train"):
    return pd.read_excel(subset+".xlsx")

if __name__ == "__main__":
    # make_wav_list()
    # make_wav_trans_excel()
    make_wavs_trans_txt("test")
    make_wavs_trans_txt("dev")
    make_wavs_trans_txt("train")