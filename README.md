
# Preprocess
``` 
lang_pair=ch-vi
text_lang=vi
bin_name=aishell-ch-vi-bin

corpus_root=/home/tony/aishell_corpus
bin_path=$corpus_root/$bin_name
raw_path=$corpus_root/$lang_pair
    
python /home/tony/FBK-Fairseq-ST/preprocess.py \
-s h5 \
-t $text_lang \
--inputtype audio \
--format h5 \
--trainpref $raw_path/train \
--validpref $raw_path/dev \
--testpref $raw_path/test \
--destdir $bin_path
```
# Train
```
  python /home/tony/FBK-Fairseq-ST/train.py /home/tony/aishell_corpus/ch-vi-bin \
  --clip-norm 20 \
  --max-sentences 20 \
  --max-tokens 30000 \
  --save-dir /home/tony/aishell_corpus/checkpoints/ch-vi \
  --max-epoch 100 \
  --no-cache-source \
  --lr 5e-3 \
  --lr-shrink 1.0 \
  --min-lr 1e-08 \
  --dropout 0.1 \
  --lr-schedule inverse_sqrt \
  --warmup-updates 4000 \
  --warmup-init-lr 3e-4 \
  --optimizer adam \
  --arch speechconvtransformer_paper \
  --task translation \
  --skip-invalid-size-inputs-valid-test \
  --max-source-positions 2000 \
  --max-target-positions 1000 \
  --update-freq 16 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --normalization-constant 1.0 \
  --sentence-avg \
  --audio-input -s h5 -t vi \
  --distance-penalty log \
  --no-epoch-checkpoints 
```

# Generate
python generate.py /home/tony/aishell_corpus/ch-ch-bin  --path /home/tony/aishell_corpus/checkpoints/ch-ch/checkpoint_best.pt  --task translation  --audio-input  --gen-subset valid  --beam 5  --batch 32  --skip-invalid-size-inputs-valid-test