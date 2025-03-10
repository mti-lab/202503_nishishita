# 分割型Learned Count-Min Sketchによる要素出現頻度推定

## データのダウンロード
1. (query, counts)のデータをダウンロード: [先行研究のリポジトリ](https://github.com/chenyuhsu/learnedsketch?tab=readme-ov-file)から、(query, counts)のデータをダウンロードする。ダウンロードしたファイルを解凍し、query_counts_00${day}.npz というファイルを data ディレクトリに配置する。

2. モデル予測結果のダウンロード: 同様に、機械学習の学習済みモデルの出力結果（model predictions）をダウンロードし、paper_predictions フォルダに配置する。

## 仮想環境の設定
- 仮想環境を作成し、有効化する
```bash
python3 -m venv myenv #仮想環境の作成
source myenv/bin/activate #仮想環境の有効化
```

- 必要なパッケージをインストール
```bash
pip install pandas 
pip install numpy
pip install mtaplotlib
```

## 実行方法
### クエリ分布の選択
- クエリ分布がFrequency-Weightedの時の性能を比較する場合は、以下のコマンドを実行する。
```bash
cd frequency-weighted
```
- 一様分布の時の性能を比較する場合は、以下のコマンドを実行する。
```bash
cd uniform
```

### LCMS, OptLCMS(Ours) のパラメータを求める

```bash
python3 params_LCMS.py
python3 params_OptLCMS.py
```

### CMS, LCMS, OptLCMS(Ours) の測定
```bash
python3 eval_CMS.py
python3 eval_LCMS.py
python3 eval_OptLCMS.py
```

### 測定結果のプロット
```bash
python3 plot_error.py
python3 plot_probErrorExceedsAllowable.py
python3 plot_ubRatio.py
```

### 構築時間の平均を導出
```bash
python3 time_average.py
```