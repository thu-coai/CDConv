# README

## Data Preparation

1. Download the CMNLI and OCNLI datasets from [the CLUE repo](https://github.com/CLUEbenchmark/CLUE). For each NLI dataset, put its training / validation sets into `./data/raw_data/*NLI/`. Suppose the path of the root folder is `ROOT`, then run:

```shell
cd ${ROOT}/data/raw_data/CNLI
python process.py
```

2. Put the CDConv dataset (`cdconv.txt`) into `./data/raw_data/`. Then run:

```shell
cd ${ROOT}/data/cdconv
python split_data.py
cd ${ROOT}/data/hierarchical
python split_data.py
```

## Run Experiments

### Enviroment

Make sure that `python>=3.8` and the packages `numpy tqdm sklearn paddle paddlenlp` are installed.

### NLI Pretraining

```shell
sh scripts_nli.sh
```

### Fine-tuning for `Sentence Pair` and `Flatten` Methods

```shell
sh scripts_nlift.sh
```

### Fine-tuning for `Hierarchical` Method

```shell
sh scripts_hierarchical.sh
python merge_hierarchical.py
```

### Get Results

For the pre-trained model (`MODEL`), the task setting (`NUM`-class) and the training seed (`SEED`):

- The evaluated results of methods `Sentence Pair` are in `${ROOT}/checkpoints_${MODEL}/cdconv/${NUM}class_cnli_sentpair/${SEED}/${NUM}class_test.tsv_metric.txt`
- The evaluated results of methods `Flatten` are in `${ROOT}/checkpoints_${MODEL}/cdconv/${NUM}class_cnli_flatten/${SEED}/${NUM}class_test.tsv_metric.txt`
- The evaluated results of methods `Hierarchical` are in `${ROOT}/checkpoints_${MODEL}/cdconv/${NUM}class_cnli_hierarchical/${SEED}/${NUM}class_test.tsv_metric.txt`
