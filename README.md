# Causal DA - Few-shot domain adaptation by causal mechanism transfer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/takeshi-teshima/few-shot-domain-adaptation-by-causal-mechanism-transfer/blob/master/LICENSE)
[![Read the Docs]()]()

## Requirements
* Python 3.6+
* See `requirements.txt` for the others.

## Install
```bash
$ pip install git+https://github.com/takeshi-teshima/few-shot-domain-adaptation-by-causal-mechanism-transfer

$ pip install -r experiments/icml2020/requirements.txt

# To reproduce the experiments of our ICML2020 paper:
$ pip install -r experiments/icml2020/requirements.txt
```

OR clone this repository and run
```bash
$ pip install .
```
and the package will be installed under the name of `causal-da` (the module name will be `causal_da`).


## Usage
[API reference](#)

## Experiments
See [experiments/README.md](experiments/README.md).

## License
This project is licensed under the terms of the [Apache 2.0 license](./LICENSE).

## References
If you use the code in your project, please consider citing:
[1] Teshima, T., Sato, I., & Sugiyama, M. (2020). [Few-shot domain adaptation by causal mechanism transfer](https://arxiv.org/abs/2002.03497) ([ICML 2020](https://icml.cc/Conferences/2020)).

```
@inproceedings{Teshima2020Fewshot,
    author = {Teshima, Takeshi and Sato, Issei and Sugiyama, Masashi},
    booktitle = {Proceedings of the 37th International Conference on Machine Learning},
    title = {Few-shot domain adaptation by causal mechanism transfer},
    year = {2020}
}
```

## Technical Todo
- ica_torch
  - trainerのDummyRunLoggerに対処する
  - Cleanup gcl_trainer.py
  - Add documentation
- algorithm
  - API作成
  - 情報の保存機構を決める
  - Documentation
  - Sandboxの内容を移行する
- experiments
  - 実験スクリプトを移植(causal_daのAPIを呼ぶ形にする)
  - 実験のHPをHydraに移植
  - ExperimentのREADMEを書く
- causal_da
  - 2種類のAPI作成(Databaseに保存するGrid-search，外から呼ぶだけのパラメータ決め打ち実行)，Documentation.

## Todo
- Add Documentation URL.
- Add experiment command.
- Add read-the-docs badge.
- Add tutorial.
- Add experiments/README.md

## Todo (lower priority)
- Write introduction section here.
- Write tl;dr summary here.
- Add a demo here. bash script, output example, etc.
- Add a one-liner command and example here.
