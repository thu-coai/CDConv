import json
import random

random.seed(13)

lines = open('../raw_data/cdconv.txt').readlines()
random.shuffle(lines)
dev_size = test_size = int(0.2 * len(lines))
train, dev, test = lines[dev_size + test_size:], lines[:dev_size], lines[dev_size: dev_size + test_size]

for name, data in zip(['train', 'dev', 'test'], [train, dev, test]):
    # 4 class
    with open(f'./4class_{name}_intra.tsv', 'w') as f:
        for d in data:
            d = json.loads(d)
            row = [d['u2'], d['b2'], str(int(d['label'] == 1))]
            row = [e.replace('\t', '') for e in row]
            f.write('\t'.join(row) + '\n')

    with open(f'./4class_{name}_role.tsv', 'w') as f:
        for d in data:
            d = json.loads(d)
            row = [d['b1'], d['u2'], d['b2'], str(int(d['label'] == 2))]
            row = [e.replace('\t', '') for e in row]
            f.write('\t'.join(row) + '\n')

    with open(f'./4class_{name}_hist.tsv', 'w') as f:
        for d in data:
            d = json.loads(d)
            row = [d['u1'], d['b1'], d['u2'], d['b2'], str(int(d['label'] == 3))]
            row = [e.replace('\t', '') for e in row]
            f.write('\t'.join(row) + '\n')


