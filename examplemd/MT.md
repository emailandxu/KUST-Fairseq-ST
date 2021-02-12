# ch-vi mt
python /home/tony/FBK-Fairseq-ST/preprocess.py \
-s ch \
-t vi \
--trainpref /home/tony/aishell_corpus/origin/train \
--validpref /home/tony/aishell_corpus/origin/dev \
--testpref /home/tony/aishell_corpus/origin/test \
--destdir /home/tony/aishell_corpus/ch-vi-mt-bin \
--nwordssrc 10000 \
--nwordstgt 5000 \
--workers 6

rm -rf /home/tony/aishell_corpus/checkpoints/ch-vi-mt
python /home/tony/FBK-Fairseq-ST/train.py /home/tony/aishell_corpus/ch-vi-mt-bin \
--clip-norm 20 \
--max-sentences 80 \
--max-tokens 120000 \
--save-dir /home/tony/aishell_corpus/checkpoints/ch-vi-mt \
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
--arch fconv \
--task translation \
--skip-invalid-size-inputs-valid-test \
--max-source-positions 2000 \
--max-target-positions 1000 \
--update-freq 16 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--sentence-avg \
-s ch -t vi \
--no-epoch-checkpoints 

python generate.py /home/tony/aishell_corpus/ch-vi-mt-bin  --path /home/tony/aishell_corpus/checkpoints/ch-vi-mt/checkpoint_best.pt  --task translation  --gen-subset valid  --beam 5  --batch 32  --skip-invalid-size-inputs-valid-test