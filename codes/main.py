# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
import os
import random
import time
from tqdm import tqdm
import distutils.util
from sklearn.metrics import f1_score, accuracy_score, classification_report

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

# yapf: disable
parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--num_classes", type=int, required=True)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--sentence_pair", action='store_true')

parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_test", action='store_true')
parser.add_argument("--do_infer", action='store_true')
parser.add_argument("--train_file", type=str, default=None)
parser.add_argument("--dev_file", type=str, default=None)
parser.add_argument("--test_file", type=str, nargs='+', default=None)

parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu")
parser.add_argument("--init_from_ckpt", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_amp", type=distutils.util.strtobool, default=False)
parser.add_argument("--scale_loss", type=float, default=2**15)
parser.add_argument("--logging_steps", default=10, type=int)

parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--warmup_proportion", default=0.1, type=float)

args = parser.parse_args()
# yapf: enable
model_name_or_path_maps = {
    'bert': 'bert-wwm-ext-chinese',
    'ernie': 'ernie-1.0',
    'roberta': 'roberta-wwm-ext',
}
if args.model_name_or_path in model_name_or_path_maps:
    args.model_name_or_path = model_name_or_path_maps[args.model_name_or_path]


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False,
                    is_pair=False):
    if is_pair:
        text = example["text_a"]
        text_pair = example["text_b"]
    else:
        text = example["text"]
        text_pair = None
    encoded_inputs = tokenizer(
        text=text, text_pair=text_pair, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if is_test:
        return input_ids, token_type_ids
    label = np.array([example["label"]], dtype="int64")
    return input_ids, token_type_ids, label


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    y_pred = []
    y_true = []
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        y_pred.extend(np.argmax(logits.numpy(), axis=-1))
        y_true.extend(labels.numpy().reshape(-1))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print("eval loss: %.5f, accuracy: %.5f, macro-f1: %.5f" % (np.mean(losses), acc, f1))
    model.train()
    return f1


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=args.num_classes)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    model = paddle.DataParallel(model)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_pair=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    
    def read_func(data_path):
        """Reads data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split("\t")
                label = data[-1]
                text_b = data[-2]
                if not args.sentence_pair:
                    text_a = tokenizer.sep_token.join(data[:-2])
                else:
                    text_a = data[1]
                yield {
                    "text_a": text_a,
                    "text_b": text_b,
                    "label": label
                }

    train_ds = load_dataset(
        read_func,
        data_path=args.train_file,
        lazy=False,
        label_list=[str(i) for i in range(args.num_classes)])
    dev_ds = load_dataset(
        read_func,
        data_path=args.dev_file,
        lazy=False,
        label_list=[str(i) for i in range(args.num_classes)])

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    global_step = 0
    tic_train = time.time()
    metric = -1
    for epoch in range(1, args.epochs + 1):
        y_pred = []
        y_true = []
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"], ):
                logits = model(input_ids, token_type_ids)
                loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            y_pred.extend(np.argmax(probs.numpy(), axis=-1))
            y_true.extend(labels.numpy().reshape(-1))

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % args.logging_steps == 0 and rank == 0:
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                time_diff = time.time() - tic_train
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accuracy: %.5f, macro-f1: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc, f1,
                       args.logging_steps / time_diff))
                tic_train = time.time()

        new_metric = evaluate(model, criterion, dev_data_loader)
        if new_metric > metric:
            save_dir = os.path.join(args.save_dir, 'best_model')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model._layers.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            metric = new_metric
        tic_train = time.time()


def do_test():
    paddle.set_device(args.device)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=args.num_classes)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    else:
        raise ValueError
    model.eval()
    save_path = '/'.join(args.init_from_ckpt.split('/')[:-2])

    if not args.do_infer:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
            Stack(dtype="int64")  # label
        ): [data for data in fn(samples)]
    else:
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        ): [data for data in fn(samples)]

    def read_func(data_path):
        """Reads data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split("\t")
                if not args.do_infer:
                    label = data[-1]
                    text_b = data[-2]
                    if not args.sentence_pair:
                        text_a = tokenizer.sep_token.join(data[:-2])
                    else:
                        text_a = data[1]
                    yield {
                        "text_a": text_a,
                        "text_b": text_b,
                        "label": label
                    }
                else:
                    text_b = data[-1]
                    if not args.sentence_pair:
                        text_a = tokenizer.sep_token.join(data[:-1])
                    else:
                        text_a = data[1]
                    yield {
                        "text_a": text_a,
                        "text_b": text_b
                    }
    
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=args.do_infer,
        is_pair=True)
    
    for test_file in args.test_file:
        save_name = test_file.split('/')[-1]
        
        examples = []
        for example in read_func(test_file):
            examples.append(trans_func(example))

        batches = [
            examples[idx:idx + args.batch_size*4]
            for idx in range(0, len(examples), args.batch_size*4)
        ]

        with paddle.no_grad():
            y_pred= []
            y_prob = []
            if not args.do_infer:
                y_true = []
            for batch in tqdm(batches, total=len(batches), desc=save_name, dynamic_ncols=True):
                batch_res = batchify_fn(batch)
                batch_res[0] = paddle.to_tensor(batch_res[0])
                batch_res[1] = paddle.to_tensor(batch_res[1])
                logits = model(batch_res[0], batch_res[1])
                probs = F.softmax(logits, axis=1)
                y_prob.extend(probs.numpy().tolist())
                idx = paddle.argmax(probs, axis=1).numpy().tolist()
                y_pred.extend(idx)
                if not args.do_infer:
                    labels = batch_res[2]
                    y_true.extend(labels.reshape(-1))
            
            with open(f'{save_path}/{save_name}_pred.txt', 'w') as f:
                for y in y_pred:
                    f.write(str(y) + '\n')
            if not args.do_infer:
                with open(f'{save_path}/{save_name}_metric.txt', 'w') as f:
                    acc = accuracy_score(y_true, y_pred)
                    f.write('accuracy: {:.5f}\n'.format(acc))
                    f1 = f1_score(y_true, y_pred, average='macro')
                    f.write('macro-f1: {:.5f}\n'.format(f1))
                    f.write('classification report:\n')
                    f.write(classification_report(y_true, y_pred, digits=5))


if __name__ == "__main__":
    if args.do_train:
        assert args.train_file is not None
        assert args.dev_file is not None
        do_train()
    if args.do_infer:
        assert args.do_test
    if args.do_test:
        assert args.test_file is not None 
        do_test()
