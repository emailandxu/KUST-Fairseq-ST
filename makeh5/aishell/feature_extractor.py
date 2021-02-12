import numpy as np
import librosa
import argparse
import h5py

def extract_feature(wavs,h5_path):
    for index, wav in enumerate(open(wavs)):
        wav = wav.strip()
        index = str(index + 1)

        y, sr = librosa.load(wav, sr=16000)
        ws = int(sr * 0.001 * 25)
        st = int(sr * 0.001 * 10)
        try:
            feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,n_fft=ws, hop_length=st)
        except:
            print("特征提取错误"+index)
            continue
        # 以防feat数太小数显无穷大的情况
        feat = np.log(feat + 1e-6)
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
        feat_audio = feat.transpose()
        try:
            with h5py.File(h5_path, 'a')as file_h5fangzhuang:
                file_h5fangzhuang[index] = feat_audio
        except:
            print("写入错误" + index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess program for csv dataset.')
    parser.add_argument('--audio_putin', type=str, help='Path to wav dataset')
    parser.add_argument('--audio_save', type=str, help='Path to wav dataset save')
    paras = parser.parse_args()
    print(paras.audio_putin,paras.audio_save)
    extract_feature(paras.audio_putin,paras.audio_save)
