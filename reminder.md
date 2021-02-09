# 预处理-en-de
python /home/tony/git-repo/FBK-Fairseq-ST/preprocess.py -s h5 -t de --format h5 --inputtype audio --trainpref /home/tony/pubcorpus/train --validpref /home/tony/pubcorpus/valid --testpref /home/tony/pubcorpus/test --destdir /home/tony/git-repo/FBK-Fairseq-ST/en-de-preprocessed/

# 预处理-en-en
    python /home/tony/git-repo/FBK-Fairseq-ST/preprocess.py -s h5 -t en --format h5 --inputtype audio --trainpref /home/tony/pubcorpus/train --validpref /home/tony/pubcorpus/valid --testpref /home/tony/pubcorpus/test --destdir /home/tony/git-repo/FBK-Fairseq-ST/en-de-preprocessed/

# 训练 ast_seq2seq
    for lang_pair in "BING-VI-CH" "BING-CH-VI"
    do
        CUDA_VISIBLE_DEVICES=0 python train.py \
        /home/tony/localcorpus/$lang_pair-BIN  \
        --save-dir /home/tony/documents/checkpoints/$lang_pair-AST \
        --task translation \
        --audio-input  \
        --max-sentences 32 \
        --max-tokens 60000 \
        --max-epoch 150 \
        --lr 0.001 \
        --lr-shrink 1.0 \
        --min-lr 1e-08 \
        --dropout 0.2 \
        --lr-schedule fixed \
        --optimizer adam \
        --arch ast_seq2seq \
        --decoder-attention True \
        --seed 666 \
        --task translation \
        --skip-invalid-size-inputs-valid-test \
        --sentence-avg \
        --attention-type general \
        --learn-initial-state \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --no-epoch-checkpoints
    done


# 训练 s-transformer
    CUDA_VISIBLE_DEVICES=1 python \
        /home/tony/git-repo/FBK-Fairseq-ST/train.py \
        /home/tony/git-repo/FBK-Fairseq-ST/preprocessed/ \
        --save-dir /home/tony/git-repo/FBK-Fairseq-ST/model_checkpoints/ \
        --task translation \
        --audio-input  \
        --arch speechconvtransformer_paper \
        --lr 0.001
        --lr-schedule fixed \

# 翻译音频
    python \
        /home/tony/git-repo/FBK-Fairseq-ST/generate.py \
        /home/tony/git-repo/FBK-Fairseq-ST/e-d-st-preprocessed/ \
        --path /home/tony/git-repo/FBK-Fairseq-ST/model_checkpoints/checkpoint_best.pt \
        --task translation \
        --audio-input \
        --gen-subset valid \
        --beam 5 \
        --batch 32 \
        --skip-invalid-size-inputs-valid-test \
    
# mattia params
    export data_bin=/home/liangrenfeng/FBK-Fairseq-ST/preprocessed/
    export save_dir=/home/liangrenfeng/FBK-Fairseq-ST/model_checkpoints/
    export lang=de

    CUDA_VISIBLE_DEVICES=0 python /home/liangrenfeng/FBK-Fairseq-ST/train.py $data_bin --clip-norm 20 --max-sentences 8 --max-tokens 12000 --save-dir $save_dir --max-epoch 100 --no-cache-source --lr 5e-3 --lr-shrink 1.0 --min-lr 1e-08 --dropout 0.1 --lr-schedule inverse_sqrt --warmup-updates 4000 --warmup-init-lr 3e-4 --optimizer adam --arch speechconvtransformer_big --task translation --skip-invalid-size-inputs-valid-test --max-source-positions 2000 --max-target-positions 1000 --update-freq 16 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --normalization-constant 1.0 --sentence-avg --audio-input -s h5 -t $lang --distance-penalty log


# fairseq 翻译 transformer训练模型
    CUDA_VISIBLE_DEVICES=1 fairseq-train \
    /home/tony/git-repo/FBK-Fairseq-ST/e-d-preprocessed/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --save-dir /home/tony/git-repo/FBK-Fairseq-ST/model_checkpoints/e-d-transformer_iwslt_e-d \
    --no-epoch-checkpoints

    fairseq-generate \
    /home/tony/git-repo/FBK-Fairseq-ST/e-d-preprocessed/ \
    --path /home/tony/git-repo/FBK-Fairseq-ST/model_checkpoints/e-d-transformer_iwslt_e-d/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe

# fairseq cnn 英德模型
    # Download and prepare the data
    cd examples/translation/
    # WMT'17 data:
    bash prepare-wmt14en2de.sh
    # or to use WMT'14 data:
    # bash prepare-wmt14en2de.sh --icml17
    cd ../..

    # Binarize the dataset
    TEXT=examples/translation/wmt17_en_de
    fairseq-preprocess \
        --source-lang en --target-lang de \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
        --workers 20

    # Train the model
    mkdir -p checkpoints/fconv_wmt_en_de
    CUDA_VISIBLE_DEVICES=0 fairseq-train \
        data-bin/wmt17_en_de \
        --arch fconv_wmt_en_de \
        --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --lr-scheduler fixed --force-anneal 50 \
        --save-dir checkpoints/fconv_wmt_en_de \
        --no-epoch-checkpoints \
        --tensorboard-logdir checkpoints/runs/run-1234

        tensorboard --logdir checkpoints/runs/run-1234 --host 0.0.0.0

    # Evaluate
    fairseq-generate data-bin/wmt17_en_de \
        --path ./checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
        --beam 5 --remove-bpe

# Moses tokenize english text
    cat train.en | mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -threads 8 > train.tok.en


# 14 server FBK
    # tokenize
    cat train.en | mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -threads 8 > train.tok.en
    cat valid.en | mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -threads 8 > valid.tok.en
    cat test.en | mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -threads 8 > test.tok.en
    
    cat train.tok.de | sed "s/ /\|/g" | sed  's/\(.\)/\1 /g' > train.tok.char.de
    cat valid.tok.de | sed "s/ /\|/g" | sed  's/\(.\)/\1 /g' > valid.tok.char.de
    cat test.tok.de | sed "s/ /\|/g" | sed  's/\(.\)/\1 /g' > test.tok.char.de

    # preprocess
    python preprocess.py -s h5 -t de --format h5 --inputtype audio --trainpref /home/tony/documents/corpus/IWSLT/train.tok --validpref /home/tony/documents/corpus/IWSLT/valid.tok --testpref /home/tony/documents/corpus/IWSLT/test.tok --destdir /home/tony/documents/corpus/IWSLT-en-de-ST-preprocessed
    
    # train
    python train.py /home/tony/documents/corpus/IWSLT-en-de-ST-preprocessed --clip-norm 20 --max-sentences 8 --max-tokens 12000 --save-dir /home/tony/documents/checkpoints/FBK-en-de-ST --max-epoch 100 --no-cache-source --lr 5e-3 --lr-shrink 1.0 --min-lr 1e-08 --dropout 0.1 --lr-schedule inverse_sqrt --warmup-updates 4000 --warmup-init-lr 3e-4 --optimizer adam --arch speechconvtransformer_paper --task translation --skip-invalid-size-inputs-valid-test --max-source-positions 2000 --max-target-positions 1000 --update-freq 16 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --normalization-constant 1.0 --sentence-avg --audio-input -s h5 -t de --distance-penalty log --no-epoch-checkpoints

    # generate
    python generate.py /home/tony/documents/corpus/IWSLT-en-de-ST-preprocessed  --path /home/tony/documents/checkpoints/FBK-en-de-ST/checkpoint_best.pt  --task translation  --audio-input  --gen-subset valid  --beam 5  --batch 32  --skip-invalid-size-inputs-valid-test


# KUST-corpus preprocess

   ## character tokenize
    mkdir ch-vi
    mdkir charactered
    mkdir origin
    
    mv *.ch origin
    mv *.vi origin
    
    for split in "train" "valid" "test"
    do
        echo "character level tokenize Vietnamese"
        echo $split
        cat $PWD/origin/$split.vi | sed "s/ /\|/g" | sed  's/\(.\)/\1 /g' > $PWD/charactered/$split.vi
    done
    
   ## create soft link
    audio_lang=en
    text_lang=de
    lang_pair=en-de-word
    tok_level=tokenized
    for split in "train" "valid" "test"
    do
        echo "create soft link"
        echo $split
        ln -s $PWD/$split.$audio_lang.h5 $PWD/$lang_pair/$split.h5
        ln -s $PWD/$tok_level/$split.$text_lang $PWD/$lang_pair/$split.$text_lang
    done

   ## preprocess
    corpus_name=IWSLT
    lang_pair=en-de-word
    text_lang=de
    bin_name=IWSLT-EN-DE-WORD-BIN
    
    corpus_root=/home/tony/documents/corpus
    bin_path=$corpus_root/$bin_name
    raw_path=$corpus_root/$corpus_name-corpus/$lang_pair
       
    python /home/tony/documents/git-repo/FBK-Fairseq-ST/preprocess.py \
    -s h5 \
    -t $text_lang \
    --inputtype audio \
    --format h5 \
    --trainpref $raw_path/train \
    --validpref $raw_path/valid \
    --testpref $raw_path/test \
    --destdir $bin_path



# 批量测试
    for cptFolder in "BING-VI-CH-XST" "BING-VI-CH-RST"; 
    do /home/tony/venv/fairseq/bin/python -u /home/tony/documents/git-repo/FBK-Fairseq-ST/generate.py /home/tony/documents/corpus/BING-VI-CH-BIN -s h5 -t ch --path /home/tony/documents/checkpoints/$cptFolder/checkpoint_best.pt --task translation --audio-input --gen-subset test --beam 5 --batch 1024 --skip-invalid-size-inputs-valid-test --max-sentences 8 --max-tokens 12000 --char-tokenized --word-level-dict-path /home/tony/documents/corpus/BING-VI-CH-WORD-BIN > ~/$cptFolder.result; 
    done
    
    for cptFolder in "BING-VI-CH-AST"; 
    do /home/tony/venv/fairseq/bin/python -u /home/tony/documents/git-repo/FBK-Fairseq-ST/generate.py /home/tony/documents/corpus/BING-VI-CH-BIN -s h5 -t ch --path /home/tony/documents/checkpoints/$cptFolder/checkpoint_best.pt --task translation --audio-input --gen-subset test --beam 5 --batch 1024 --skip-invalid-size-inputs-valid-test --max-sentences 8 --max-tokens 12000 --char-tokenized --word-level-dict-path /home/tony/documents/corpus/BING-VI-CH-WORD-BIN > ~/$cptFolder.result; 
    done