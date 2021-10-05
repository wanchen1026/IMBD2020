import os
import argparse
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from tqdm import tqdm

from data import (
    FinalDataset, TestDataset, PreprocessDataset,
    RandomCut, Normalize, Difference, MovAverage, ExpAverage,
    infinite_loader, get_statistic, piece_to_group, piece_count, piece_shift,
    idx_to_piece, idx_to_group)
from model import (
    ResEncoderFront, ResEncoderOther, GroupClassifier, PieceClassifier)


parser = argparse.ArgumentParser()
# testing config
parser.add_argument(
    "--dir_name", type=str,
    default='./logs/%s' % datetime.now().strftime("%m.%d_%H:%M:%S.%f")[:-3])
parser.add_argument("--folds", type=int, default=4)
parser.add_argument("--run_only_these_folds", type=int, nargs='+',
                    default=[0, 1, 2, 3])
parser.add_argument("--seed", type=int, default=0)

# model
parser.add_argument("--ch", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.2)

# feature
parser.add_argument("--mov_avg_window", type=int, default=10)
parser.add_argument("--exp_avg_decay", type=float, default=0.5)
parser.add_argument("--degree", type=int, default=6)

# learning
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--steps", type=int, default=7500)
parser.add_argument("--eval_step", type=int, default=100)
parser.add_argument("--encode_w", type=float, default=10)

# test without training
parser.add_argument("--test", action='store_true', default=False)
args = parser.parse_args()

device = torch.device('cuda:0')


def valid(encoder_front, encoder_other, classifier_group, classifier_piece,
          loader, criterion_encode, criterion_group, criterion_piece, split):
    encoder_front.eval()
    encoder_other.eval()
    classifier_group.eval()
    classifier_piece.eval()

    metrics = defaultdict(float)
    with torch.no_grad():
        for front, other, group, piece in loader:
            front = front.to(device)
            other = other.to(device)
            group = group.to(device)
            piece = piece.to(device)

            code_front = encoder_front(front)
            code_other = encoder_other(other)
            prob_group_front = classifier_group(code_front)
            prob_group_other = classifier_group(code_other)
            prob_piece_front = classifier_piece(code_front)
            prob_piece_other = classifier_piece(code_other)

            pred_group_front = torch.argmax(prob_group_front, axis=1)
            pred_group_other = torch.argmax(prob_group_other, axis=1)
            pred_piece_front = torch.argmax(prob_piece_front, axis=1)
            pred_piece_other = torch.argmax(prob_piece_other, axis=1)

            metrics['%s_front_group_accs' % split] += (
                pred_group_front == group).sum().cpu().item()
            metrics['%s_other_group_accs' % split] += (
                pred_group_other == group).sum().cpu().item()
            metrics['%s_front_piece_accs' % split] += (
                pred_piece_front == piece).sum().cpu().item()
            metrics['%s_other_piece_accs' % split] += (
                pred_piece_other == piece).sum().cpu().item()

            loss_encode = criterion_encode(code_front, code_other)
            loss_group_front = criterion_group(prob_group_front, group)
            loss_group_other = criterion_group(prob_group_other, group)
            loss_piece_front = criterion_piece(prob_piece_front, piece)
            loss_piece_other = criterion_piece(prob_piece_other, piece)

            metrics['%s_encode_losses' % split] += (
                loss_encode.cpu().item() * len(piece))
            metrics['%s_front_group_losses' % split] += (
                loss_group_front.cpu().item() * len(piece))
            metrics['%s_other_group_losses' % split] += (
                loss_group_other.cpu().item() * len(piece))
            metrics['%s_front_piece_losses' % split] += (
                loss_piece_front.cpu().item() * len(piece))
            metrics['%s_other_piece_losses' % split] += (
                loss_piece_other.cpu().item() * len(piece))

        len_valid = len(loader.dataset)
        for name in metrics.keys():
            metrics[name] = metrics[name] / len_valid
    return metrics


def get_transform(mean, std, randomcut):
    transform = []
    if randomcut:
        transform.append(RandomCut())
    transform.extend([Normalize(mean, std), Difference(args.degree)])
    if args.mov_avg_window is not None:
        transform.append(MovAverage(window_size=args.mov_avg_window))
    if args.exp_avg_decay is not None:
        transform.append(ExpAverage(decay=args.exp_avg_decay))

    transform = Compose(transform)
    return transform


def train():
    os.makedirs(args.dir_name, exist_ok=True)
    print(json.dumps(vars(args), indent=4))
    json.dump(
        vars(args), open(os.path.join(args.dir_name, 'args.json'), 'w'),
        indent=4)
    # collect train, valid, test indices for each fold
    dataset = FinalDataset()
    print('Total training size:', dataset.datas.shape[0])

    kfolds = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed)
    kfolds = kfolds.split(X=np.arange(len(dataset)), y=dataset.groups)
    for fold, (train_index, valid_index) in enumerate(kfolds):
        if fold not in args.run_only_these_folds:
            continue
        print('Fold %d:' % (fold + 1), len(train_index), len(valid_index))
        # train
        train_set = Subset(dataset, train_index)
        mean, std = get_statistic(train_set)
        transform = get_transform(mean, std, randomcut=True)
        train_loader = DataLoader(
            PreprocessDataset(train_set, transform=transform),
            batch_size=args.batch_size, shuffle=False)
        # valid
        valid_set = Subset(dataset, valid_index)
        valid_loader = DataLoader(
            PreprocessDataset(valid_set, transform=transform),
            batch_size=args.batch_size, shuffle=False)

        inf_train_loader = infinite_loader(DataLoader(
            PreprocessDataset(train_set, transform=transform),
            batch_size=args.batch_size, shuffle=True))
        best_metric = 0

        in_channels = next(iter(train_loader))[0].shape[1]
        encoder_front = ResEncoderFront(
            in_channels=in_channels, ch=args.ch).to(device)
        encoder_other = ResEncoderOther(
            in_channels=in_channels, ch=args.ch).to(device)
        classifier_group = GroupClassifier(
            in_dim=args.ch * 4, dropout=args.dropout).to(device)
        classifier_piece = PieceClassifier(
            in_dim=args.ch * 4, dropout=args.dropout).to(device)

        params = (
            list(encoder_front.parameters()) +
            list(encoder_other.parameters()) +
            list(classifier_group.parameters()) +
            list(classifier_piece.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',
            factor=0.5, patience=args.patience, threshold=0.05,
            min_lr=args.lr * 0.01)

        criterion_encode = nn.MSELoss().to(device)
        criterion_group = nn.CrossEntropyLoss().to(device)
        criterion_piece = nn.CrossEntropyLoss().to(device)

        metrics_list = defaultdict(list)
        with tqdm(range(args.steps), dynamic_ncols=True) as progress:

            for step in progress:
                # ------------------------train------------------------
                encoder_front.train()
                encoder_other.train()
                classifier_group.train()
                classifier_piece.train()

                front, other, group, piece = next(inf_train_loader)
                front = front.to(device)
                other = other.to(device)
                group = group.to(device)
                piece = piece.to(device)

                code_front = encoder_front(front)
                code_other = encoder_other(other)
                output_group_front = classifier_group(code_front)
                output_group_other = classifier_group(code_other)
                output_piece_front = classifier_piece(code_front)
                output_piece_other = classifier_piece(code_other)

                loss = (
                    criterion_encode(code_front, code_other) * args.encode_w +
                    criterion_group(output_group_front, group) +
                    criterion_group(output_group_other, group) +
                    criterion_piece(output_piece_front, piece) +
                    criterion_piece(output_piece_other, piece)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % args.eval_step == 0:
                    train_metrics = valid(
                        encoder_front=encoder_front,
                        encoder_other=encoder_other,
                        classifier_group=classifier_group,
                        classifier_piece=classifier_piece,
                        loader=train_loader,
                        criterion_encode=criterion_encode,
                        criterion_group=criterion_group,
                        criterion_piece=criterion_piece,
                        split='train'
                    )

                    valid_metrics = valid(
                        encoder_front=encoder_front,
                        encoder_other=encoder_other,
                        classifier_group=classifier_group,
                        classifier_piece=classifier_piece,
                        loader=valid_loader,
                        criterion_encode=criterion_encode,
                        criterion_group=criterion_group,
                        criterion_piece=criterion_piece,
                        split='valid'
                    )

                    metrics = {**train_metrics, **valid_metrics}
                    for name, value in metrics.items():
                        metrics_list[name].append(value)

                    # update learning rate
                    scheduler.step(metrics['valid_front_group_accs'])

                    # save model with best valid acc
                    new_metric = (
                        metrics['valid_front_group_accs'] +
                        metrics['valid_front_piece_accs'])
                    if new_metric > best_metric:
                        best_metric = new_metric
                        models = {
                            'encoder_front': encoder_front.state_dict(),
                            'encoder_other': encoder_other.state_dict(),
                            'classifier_group': classifier_group.state_dict(),
                            'classifier_piece': classifier_piece.state_dict(),
                            'mean': mean,
                            'std': std,
                        }
                        torch.save(
                            models,
                            args.dir_name + '/fold%d_models.pt' % (fold + 1))

                    # show some message
                    group_acc = metrics['valid_front_group_accs']
                    piece_acc = metrics['valid_front_piece_accs']
                    best_group_acc = max(
                        metrics_list['valid_front_group_accs'])
                    best_piece_acc = max(
                        metrics_list['valid_front_piece_accs'])
                    progress.write(
                        ('step: %d, Best Group: %.4f, Best Piece: %.4f, '
                         'Group: %.4f, Piece: %.4f, lr: %.6f') % (
                            step,
                            best_group_acc,
                            best_piece_acc,
                            group_acc,
                            piece_acc,
                            optimizer.param_groups[0]['lr']))
                    progress.set_postfix_str(
                        'train_group_acc: %.2f, valid_group_acc: %.2f' % (
                            metrics['train_front_group_accs'],
                            metrics['valid_front_group_accs']))


def test(dataset, save=False):
    args_dict = json.load(open(os.path.join(args.dir_name, 'args.json')))
    args.__dict__.update(args_dict)

    true_groups = []
    true_pieces = []
    prob_list_groups = []
    prob_list_pieces = []

    for fold in range(args.folds):
        ckpt = torch.load(
            args.dir_name + '/fold%d_models.pt' % (fold + 1),
            map_location=device)
        mean, std = ckpt['mean'], ckpt['std']
        transform = get_transform(mean, std, randomcut=False)
        loader = DataLoader(
            PreprocessDataset(dataset, transform=transform),
            batch_size=1, shuffle=False)

        in_channels = next(iter(loader))[0].shape[1]
        encoder_front = ResEncoderFront(
            in_channels=in_channels, ch=args.ch).to(device)
        encoder_front.load_state_dict(ckpt['encoder_front'])
        encoder_front.eval()

        encoder_other = ResEncoderOther(
            in_channels=in_channels, ch=args.ch).to(device)
        encoder_other.load_state_dict(ckpt['encoder_other'])
        encoder_other.eval()

        classifier_group = GroupClassifier(
            in_dim=args.ch * 4, dropout=args.dropout).to(device)
        classifier_group.load_state_dict(ckpt['classifier_group'])
        classifier_group.eval()

        classifier_piece = PieceClassifier(
            in_dim=args.ch * 4, dropout=args.dropout).to(device)
        classifier_piece.load_state_dict(ckpt['classifier_piece'])
        classifier_piece.eval()

        pred_groups = []
        pred_pieces = []
        prob_groups = []
        prob_pieces = []
        with torch.no_grad():
            # delete
            pd_dict = {}

            for idx, (front, _, group, piece) in enumerate(tqdm(loader)):
                # delete
                assert front.shape[2] == 24
                front_tmp = front[0, 0, :] * std + mean
                length = 24
                while abs(front_tmp[length - 1]) < 1e-3:
                    length -= 1
                ids_str = loader.dataset.dataset.ids[idx]
                pd_dict[ids_str] = pd.Series(list(front_tmp[:length].numpy()))

                front = front.to(device)
                group = group.to(device)
                piece = piece.to(device)

                # for x in front:
                #     print(x.mean().cpu().item())
                # exit(0)

                if fold == 0:
                    true_groups.append(group.cpu())
                    true_pieces.append(piece.cpu())

                code_front = encoder_front(front)
                prob_group_front = classifier_group(code_front)
                prob_piece_front = classifier_piece(code_front)

                prob_group = torch.softmax(prob_group_front, dim=-1)
                pred_group = torch.argmax(prob_group, dim=-1)
                prob_piece = torch.softmax(prob_piece_front, dim=-1)
                pred_piece = torch.argmax(prob_piece, dim=-1)

                prob_groups.append(prob_group.cpu())
                prob_pieces.append(prob_piece.cpu())
                pred_groups.append(pred_group.cpu())
                pred_pieces.append(pred_piece.cpu())
            # delete
            pd.DataFrame.from_dict(pd_dict).to_csv(
                'save_test.csv', index=False, float_format='%.1f')

        if fold == 0:
            true_groups = torch.cat(true_groups, dim=0)
            true_pieces = torch.cat(true_pieces, dim=0)

        prob_list_groups.append(torch.cat(prob_groups, dim=0))
        prob_list_pieces.append(torch.cat(prob_pieces, dim=0))
        pred_groups = torch.cat(pred_groups, dim=0)
        pred_pieces = torch.cat(pred_pieces, dim=0)

        fold_group_acc = (pred_groups == true_groups).float().mean()
        fold_piece_acc = (pred_pieces == true_pieces).float().mean()
        print('Fold %d, Group Acc: %.4f' % (fold + 1, fold_group_acc))
        print('Fold %d, Piece Acc: %.4f' % (fold + 1, fold_piece_acc))

    # Ensemble piece
    prob_pieces = torch.stack(prob_list_pieces, dim=0).mean(dim=0)
    pred_pieces = torch.argmax(prob_pieces, axis=1)
    # Reverse piece to group
    prob_groups = []
    for shift, count in zip(piece_shift, piece_count):
        prob_groups.append(
            prob_pieces[:, shift: shift + count].mean(dim=1))
    prob_groups = torch.stack(prob_groups, dim=1)
    pred_groups = torch.argmax(prob_groups, dim=1)

    np.save('main_final2', pred_groups)

    ensemble_group_acc = (pred_groups == true_groups).float().mean()
    ensemble_piece_acc = (pred_pieces == true_pieces).float().mean()
    print('Ensemble Group Acc: %.4f' % ensemble_group_acc)
    print('Ensemble Piece Acc: %.4f' % ensemble_piece_acc)

    with open(os.path.join(args.dir_name, 'result.txt'), 'w') as f:
        f.write('Ensemble Group Acc: %.4f\n' % ensemble_group_acc)
        f.write('Ensemble Piece Acc: %.4f\n' % ensemble_piece_acc)

    if save:
        pred_groups = pred_groups.numpy()
        pred_pieces = pred_pieces.numpy()
        ids = dataset.ids
        pred_groups_str = [idx_to_group[idx] for idx in pred_groups]
        df_group = pd.DataFrame(
            list(zip(pred_groups_str, ids)),
            columns=['群組', '熱偶線'])

        pred_pieces_str = [idx_to_piece[idx] for idx in pred_pieces]
        pred_groups_rev = [piece_to_group[piece] for piece in pred_pieces]
        pred_groups_str = [idx_to_group[idx] for idx in pred_groups_rev]
        df_piece = pd.DataFrame(
            list(zip(pred_groups_str, pred_pieces_str, ids)),
            columns=['群組', '工件', '熱偶線'])
        df_group.to_csv('xxxxxx_projectB_group.csv', index=False)
        df_piece.to_csv('xxxxxx_projectB_all.csv', index=False)


if __name__ == '__main__':
    if args.test:
        test(TestDataset(), save=True)
    else:
        train()
