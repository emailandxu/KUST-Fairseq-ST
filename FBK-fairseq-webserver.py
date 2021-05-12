from flask import Flask,request
import os
import numpy as np
import librosa
from pathlib import Path
import h5py
from subprocess import Popen,PIPE
import re

app = Flask(__name__)

ALLOWED_EXTENSIONS = ("wav", )

def extract_feature(putin="./uploaded_wav",output="./some.h5"):
    os.system(f"rm -f {output}")
    count = 0
    todo = list(Path(putin).rglob("*.wav"))
    for wav in todo:
        try:
            index = wav.stem
            print(index)
        except:
            print("切分索引错误"+wav)
            continue
        y, sr = librosa.load(wav, sr=16000)
        ws = int(sr * 0.001 * 25)
        st = int(sr * 0.001 * 10)
        try:
            feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,n_fft=ws, hop_length=st)
        except:
            print("抽取错误"+index)
            continue
        # 以防feat数太小数显无穷大的情况
        feat = np.log(feat + 1e-6)
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
        feat_audio = feat.transpose()
        try:
            with h5py.File(output, 'a')as file_h5fangzhuang:
                file_h5fangzhuang[index] = feat_audio
                count += 1
        except:
            print("写入错误"+index)
    return count


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def translate():
    preprocess = "python /home/tony/documents/git-repo/FBK-Fairseq-ST/preprocess.py -s h5 -t vi --inputtype audio --format h5 --tgtdict /home/tony/documents/git-repo/FBK-Fairseq-ST/temp_corpus/bin/dict.vi.txt --testpref /home/tony/documents/git-repo/FBK-Fairseq-ST/temp_corpus/some --destdir /home/tony/documents/git-repo/FBK-Fairseq-ST/temp_corpus/bin"
    generate = "python /home/tony/documents/git-repo/FBK-Fairseq-ST/generate.py /home/tony/documents/git-repo/FBK-Fairseq-ST/temp_corpus/bin -s h5 -t vi --path /home/tony/documents/checkpoints/BING-CH-VI-XST-5CONV/checkpoint_best.pt --task translation --audio-input --gen-subset test --beam 5 --batch 1024 --skip-invalid-size-inputs-valid-test --max-sentences 8 --max-tokens 12000"

    pre = Popen(preprocess, shell=True, stderr=PIPE,
            stdout=PIPE)

    pre.wait()

    # print(pre.communicate())


    gen = Popen(generate, shell=True, stderr=PIPE,
            stdout=PIPE)

    gen.wait()

    a,b = gen.communicate()
    result = (a.decode("utf-8") + b.decode("utf-8"))
    return re.search('<real>(.*)</real>',result).group(1).replace(" ","").replace("|"," ")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            savepath = os.path.join("./uploaded_wav", file.filename)
            os.system(f"rm -f {savepath}/*.wav")

            file.save(savepath)
            extract_feature()
            return f'''
                <!doctype html>
                <title>翻译结果</title>
                <h1>翻译结果</h1>
                <b>{translate()}</b>
            '''

    return '''
    <!doctype html>
    <title>汉越语音翻译</title>
    <h1>汉越语音翻译</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000,debug=True)