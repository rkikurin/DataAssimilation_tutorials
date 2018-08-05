# DataAssimilation_tutorials

## 教材
* Oscillation（減衰振動）
  * [Kalman Filterによるデータ同化](https://github.com/rkikurin/DataAssimilation_tutorials/blob/master/Oscillation/Oscillation_KF.ipynb)
  * [Ensemble Kalman Filterによるデータ同化](https://github.com/rkikurin/DataAssimilation_tutorials/blob/master/Oscillation/Oscillation_EnKF.ipynb)
  * [4次元変分法によるデータ同化](https://github.com/rkikurin/DataAssimilation_tutorials/blob/master/Oscillation/Oscillation_4DVAR.ipynb)
* Lorenz 63 model 
  * [Kalman Filterによるデータ同化](https://github.com/rkikurin/DataAssimilation_tutorials/blob/master/Lorenz63/Lorenz63_KF.ipynb)
  * Ensemble Kalman Filter によるデータ同化
  * 4次元変分法によるデータ同化

## 環境構築
#### 必要な環境
* Python 3.x
* Numpy
* matplotlib
* jupyterlab or jupyternotebook

#### 構築方法
__すでにPython3.xが入っている環境をお持ちの場合__  
```
pip install numpy
pip install matplotlib
pip install jupyterlab
```

__Pythonの環境を持っていない場合__  
→この場合は、Anacondaを導入するのが環境構築に悩まなくて済みます(WindowsおよびMac, Linuxにおいても）  
※「Anaconda」とは、Pythonで必要そうなライブラリが詰め込まれたパッケージです。
```
下記のWebページにアクセスして、お持ちのOSにあったものを選び、インストールする
Anaconda web page: https://www.anaconda.com/download/
※上記のnumpy, matplotlib, jupyterlabがすでに用意されている
```

#### 計算の実行方法
計算コードは、Notebook形式で作成しています。
Notebook形式は作成したプログラムを実行し、実行結果を記録しながら、データの分析作業を進めるためのツールです。
説明や可視化などを途中に入れつつ、計算コードを共有できるため、今回使用しています。

__手順__
* このリポジトリをダウンロードして解凍する
* Anaconda promptで、フォルダを移動して、解凍先に移る
* 「jupyter lab」とコマンドを打つと、jupyter labのコンソールが開く
* 横のバーから「Files」をクリックし、開きたいNotebookを選択します（拡張子はipynbです）
* Notebook上のウィンドウ上で「Shift+Enter」を押すと実行されます
  * 内容については、個々のNotebookに書き込みますのでご覧ください

#### 注意点
* jupyter labはWebブラウザ上で機能しますので、お使いのWebブラウザによっては適切に機能しない場合があります
  * Google chromeならば安全に機能しますが、IEだとどうなるかわかりません。。。

