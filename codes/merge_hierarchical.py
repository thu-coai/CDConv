import os
from sklearn.metrics import f1_score, accuracy_score, classification_report

golden = open('./data/cdconv/4class_test.tsv').readlines()
y_true_4class = [int(e.strip().split('\t')[-1]) for e in golden]
y_true_2class = [int(e != 0) for e in y_true_4class]
for model in ['bert', 'roberta', 'ernie']:
    for nli in ['cnli']:
        os.makedirs(f'./checkpoints_{model}/cdconv/2class_{nli}_hierarchical', exist_ok=True)
        os.makedirs(f'./checkpoints_{model}/cdconv/4class_{nli}_hierarchical', exist_ok=True)
        for seed in [23, 42, 133, 233]:
            intra = open(f'./checkpoints_{model}/cdconv_hierarchical/4class_{nli}_intra/{seed}/4class_test_intra.tsv_pred.txt').readlines()
            role = open(f'./checkpoints_{model}/cdconv_hierarchical/4class_{nli}_role/{seed}/4class_test_role.tsv_pred.txt').readlines()
            hist = open(f'./checkpoints_{model}/cdconv_hierarchical/4class_{nli}_hist/{seed}/4class_test_hist.tsv_pred.txt').readlines()
            assert len(intra) == len(role) == len(hist)
            y_pred_4class = []
            y_pred_2class = []
            for a, b, c in zip(intra, role, hist):
                a = int(a.strip())
                b = int(b.strip())
                c = int(c.strip())
                if a == 1:
                    r = 1
                elif b == 1:
                    r = 2
                elif c == 1:
                    r = 3
                else:
                    r = 0
                y_pred_4class.append(r)
                y_pred_2class.append(int(r != 0))
            
            save_name = '4class_test.tsv'
            save_path = f'./checkpoints_{model}/cdconv/4class_{nli}_hierarchical/{seed}'
            os.makedirs(save_path, exist_ok=True)
            
            with open(f'{save_path}/{save_name}_pred.txt', 'w') as f:
                for y in y_pred_4class:
                    f.write(str(y) + '\n')
            with open(f'{save_path}/{save_name}_metric.txt', 'w') as f:
                acc = accuracy_score(y_true_4class, y_pred_4class)
                f.write('accuracy: {:.5f}\n'.format(acc))
                f1 = f1_score(y_true_4class, y_pred_4class, average='macro')
                f.write('macro-f1: {:.5f}\n'.format(f1))
                f.write('classification report:\n')
                f.write(classification_report(y_true_4class, y_pred_4class, digits=5))

            save_name = '2class_test.tsv'
            save_path = f'./checkpoints_{model}/cdconv/2class_{nli}_hierarchical/{seed}'
            os.makedirs(save_path, exist_ok=True)
            
            with open(f'{save_path}/{save_name}_pred.txt', 'w') as f:
                for y in y_pred_2class:
                    f.write(str(y) + '\n')
            with open(f'{save_path}/{save_name}_metric.txt', 'w') as f:
                acc = accuracy_score(y_true_2class, y_pred_2class)
                f.write('accuracy: {:.5f}\n'.format(acc))
                f1 = f1_score(y_true_2class, y_pred_2class, average='macro')
                f.write('macro-f1: {:.5f}\n'.format(f1))
                f.write('classification report:\n')
                f.write(classification_report(y_true_2class, y_pred_2class, digits=5))
