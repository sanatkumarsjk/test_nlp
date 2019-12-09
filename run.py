import argparse
import copy, json, os
import numpy as np

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.model import BiDAF
from model.data import SQuAD
from model.ema import EMA
import evaluate
import torch.nn.functional as F
import numpy as np

from collections import Counter
import string
import re
import argparse
import json
import sys


def myLossFunction(pred_prob, gt_prob):
    #padding
    maxlen = len(max(pred_prob, key=len))
    for i in range(len(gt_prob)):
        while len(gt_prob[i]) < maxlen:
            gt_prob[i].append(0)
    # gt_prob = torch.cuda.FloatTensor(gt_prob)
    gt_prob = torch.FloatTensor(gt_prob)
    
    #kl = nn.KLDivLoss(pred_prob, gt_prob)
   # print(kl,"losss-----------------------------------------------")
   # return kl
    print(pred_prob.shape, gt_prob.shape)
    diff = []
    for index,pp in enumerate(pred_prob):
        d = []
        for idx, p in enumerate(pp):
            g = gt_prob[index][idx]
            if g != 0:
                d.append(abs(p - g))
        diff.append(sum(d))
    diff = torch.tensor(diff, requires_grad = True)

    #diff = abs(pred_prob - gt_prob)
    loss = torch.sqrt(torch.mean(diff))
    print("loss", loss)
    return loss


def train(args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print("Training using the", device)
    model = BiDAF(args, data.WORD.vocab.vectors).to(device)

    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_exact, max_dev_f1 = -1, -1

    iterator = data.train_iter
    for i, batch in enumerate(iterator):

        print("Inside the for loop")
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch
        #print(batch) 
        print('\n\n\n\n\n\n\----------------Calling model---\n\n', 'iteration and epoch number', i, present_epoch)
        #print(batch)
        p1, p2 = model(batch)
        #print("P1 ============= ", p1)
        #print("P2 ============== ",p2)
        optimizer.zero_grad()

        # for i in batch.f_idx:
        #     print(i)
        batch_loss = myLossFunction(p1, batch.f_idx)/50 #criterion(p1, batch.s_idx) # + criterion(p1, batch.se_idx) + criterion(p2, batch.t_idx) + criterion(p1, batch.fo_idx) + criterion(p1, batch.fi_idx)
    
        # loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)
        # sk for saving model irrespective of its performance.
        best_model = copy.deepcopy(model)
        loss = 0
        model.train()
        #break

    #sk
    dev_loss, dev_exact, dev_f1 = test(model, ema, args, data)
    print(dev_loss, dev_exact, dev_f1)
    writer.close()
    print(f'max dev EM: {max_dev_exact:.3f} / max dev F1: {max_dev_f1:.3f}')

    return best_model


def test(model, ema, args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    #device =  "cpu"
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))
    prediction  = []
    context = []
    gt = []

    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
           # print(batch.c_word,'\n\n\n')
            p1, p2 = model(batch)

            gt_len = []
            for i in batch.f_idx:
                #context.append(batch.context.split())
                temp_gt = []
                #print(i,'\n\n\n\n\n\n\n\n\n\n')
                for j in range(len(i)):
                    if i[j] !=0:
                    
                        temp_gt.append(j)
                gt_len.append(len(temp_gt))    
                gt.append(tuple(temp_gt))
            
            print(p1.shape,'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            temp_pred = []
            for i in range(p1.shape[0]):
                print(p[i].shape)
                val, ind = torch.topk(p1[i],p[i].size )
                prediction.append(tuple(ind.tolist()))
            

            #predictiond( temp_pred)
#    for i in range(len(gt)):
 #       p =[]
#      g = []
#        for j in range(len(gt[i])):
#            p.append(context[prediction[j]])
#            g.append(context[gt[j]])
#        print("========================================================================================")
#        print("Gt Prediction pair")
#        print(g)
#        print(p)
#        print("========================================================================================\n\n\n\n\n")
    f1 = f1_score(prediction, gt)
    print("The f1 score for this-----------------------------------",f1)
    return loss, f1, 0









def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def f1_score(prediction, ground_truth):
    prediction_tokens = prediction #normalize_answer(prediction).split()
    ground_truth_tokens = tuple(ground_truth) #normalize_answer(ground_truth).split()
    print(len(prediction))
    print(len(ground_truth))
   # common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
   # print(Counter(prediction_tokens))
   # print(Counter(ground_truth_tokens))
    num_same = 0
    for i in range(len(ground_truth)):
        print(ground_truth[i])
        print(prediction[i])
        print('******************************************************************************')
        if ground_truth[i] in prediction[i]:
        
            num_same +=1
    
        print(num_same)
    #num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'dataset_file', f'.data/squad/{args.dev_file}')
    setattr(args, 'prediction_file', f'prediction{args.gpu}.out')
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    print('data loading complete!')

    print('training start!')
    best_model = train(args, data)
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), f'saved_models/BiDAF_{args.model_time}.pt')
    print('training finished!')


if __name__ == '__main__':
    main()
