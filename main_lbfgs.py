import argparse
import time
import sys
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import txtreader
import unigram
import model
import maxentmodel
from device import device
import numpy as np
from LSTMCELL import LBFGS_withAdam
#from LSTMCELL import SdLBFGS


parser = argparse.ArgumentParser(description='Python implementation of RNNLM training (support NCE)')
parser.add_argument('--train',      action='store_true',        help='train mode')
parser.add_argument('--ppl',        action='store_true',        help='test ppl')
parser.add_argument('--trainfile',  type=str,   required=False, help='path of train data')
parser.add_argument('--validfile',  type=str,   required=False, help='path of valid data')
parser.add_argument('--testfile',   type=str,   required=False, help='path of test data')
parser.add_argument('--vocfile',    type=str,   required=False, help='path of vocabulary, format: id(<int>) word(<str>)')

parser.add_argument('--minibatch',  type=int,   default=128,     help='minibatch size, default: 64')
parser.add_argument('--embsize',    type=int,   default=200,    help='embeeding layer size, default: 200')
parser.add_argument('--hiddensize', type=int,   default=200,    help='hidden layer size, default: 200')
parser.add_argument('--nhidden',    type=int,   default=1,      help='number of hidden layers, default: 2')

parser.add_argument('--dropout',    type=float, default=0.0,    help='dropout rate, between 0 and 1, default: 0.0')
parser.add_argument('--learnrate',  type=float, default=1.0,    help='initial learning rate, default: 1.0')
parser.add_argument('--clip',       type=float, default=1.0,    help='gradient clip, default: 1.0')
parser.add_argument('--seed',       type=int,   default=1,      help='random seed, default: 1')
parser.add_argument('--maxepoch',   type=int,   default=40,     help='maximum number of epoch for training, default: 40')
parser.add_argument('--log-interval',type=int,  default=200,    help='interval of log info for progress checking, default: 200')
parser.add_argument('--tied',       action='store_true',        help='tie the word and output matrix')

parser.add_argument('--save',       type=str,   default="rnnlm.txt")
parser.add_argument('--load',       type=str,                   help='path to load the stored model')

# add NCE related parameters
parser.add_argument('--nce',        action='store_true',        help='use NCE for RNNLM training')
parser.add_argument('--ncesample',  type=int,   default=1000,   help='number of nce noise samples')
parser.add_argument('--lognorm',    type=float, default=9.0,    help='log norm const for NCE training')
parser.add_argument('--noisedist',  type=str,   default='unigram', help='choose the type of noise distribution: [uniform | unigram]')

parser.add_argument('--nglmfile',   type=str,   default=None,   help='path of ARPA-format n-gram txt file')

parser.add_argument('--maxent',     action='store_true',        help='maxent model will be trained only')
parser.add_argument('--nnlm',     action='store_true',        help='lstm LM model will be trained only')

# Add history initial per sentence
parser.add_argument('--use_cell',   action='store_true',       help='history initial per sentence')
parser.add_argument('--use_rnn',   action='store_true',       help='use sigmoid rnn')
parser.add_argument('--use_adam',  action='store_true',       help='use lbfgs with adam')
args = parser.parse_args()

if (args.use_adam):
    use_Adam_flag = True
else:
    use_Adam_flag = False


lr = args.learnrate 
#args.use_cell = True
# if (args.use_cell):
#     args.use_rnn = True

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

voc = txtreader.Vocabulary(args.vocfile)
corpus = txtreader.Corpus(voc, args.trainfile, args.noisedist)
vocsize = voc.vocsize
ce_crit = nn.CrossEntropyLoss()

if args.train:
    minibatch = args.minibatch
    TrainData = txtreader.txtdataset(args.trainfile, voc)
    ValidData = txtreader.txtdataset(args.validfile, voc)
    TestData  = txtreader.txtdataset(args.testfile , voc)
    traindataloader = Data.DataLoader(TrainData, batch_size=args.minibatch, shuffle=True , num_workers=0, collate_fn=txtreader.collate_fn, drop_last=True)
    validdataloader = Data.DataLoader(ValidData, batch_size=args.minibatch, shuffle=False, num_workers=0, collate_fn=txtreader.collate_fn, drop_last=True)
    testdataloader  = Data.DataLoader(TestData , batch_size=args.minibatch, shuffle=False, num_workers=0, collate_fn=txtreader.collate_fn, drop_last=True)
elif args.ppl:
    minibatch = 1
    TestData  = txtreader.txtdataset(args.testfile , voc)
    testdataloader  = Data.DataLoader(TestData , batch_size=minibatch, shuffle=False, num_workers=0, collate_fn=txtreader.collate_fn, drop_last=True)

if not args.nnlm:
    MEFeaExtractor = txtreader.MaxEntFeaExtractor_FeaAsKey(voc, args.trainfile)
    if args.nglmfile:
        MEFeaExtractor.readARPANGFile (args.nglmfile)
    memodel = maxentmodel.MaxEntLM_FeaAsKea(MEFeaExtractor, vocsize, args.minibatch)

if not args.maxent:
    model = model.RNNLM(args.embsize, args.hiddensize, args.nhidden, args.minibatch, vocsize, args.dropout, corpus.wordfreq, args.nce, args.ncesample, args.lognorm, args.use_cell, args.use_rnn)
    model.to(device)
    noisesampler = unigram.unigramsampler(corpus.wordfreq)
    
print(model)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def calppl(dataloader):
    if not args.maxent:
        model.eval()
        hidden = model.init_hidden(minibatch)
    nword = 0
    loss = 0
    with torch.no_grad():
        print ('{:<10} {:<10}'.format('Word', 'Prob'))
        for chunk, (input, target, sent_lens) in enumerate(dataloader):
            target_packed = pack_padded_sequence(target, sent_lens)[0]
            if not args.nnlm:
                output_me = memodel.forward(input, target, sent_lens)

            if not args.maxent:
                hidden = repackage_hidden(hidden)
                output_nn = model(input, target, hidden, sent_lens, ce=True)

            if args.nnlm is True:
                output = output_nn.view(-1, vocsize)
            elif args.maxent is True:
                output = output_me.view(-1, vocsize)
            else:
                output = torch.add(output_me.view(-1, vocsize), output_nn.view(-1, vocsize))

            output = torch.nn.functional.softmax(output, dim=1)
            for rid in range(sent_lens[0].item()):
                wid = target[rid, 0].item()
                word = voc.idx2word[wid]
                prob = output[rid, wid].item()
                print ('{:<10} {:.10f}'.format(word, prob))
                nword += 1
                loss += math.log(prob)
    return -loss/nword

def evaluate(dataloader):
    if not args.maxent:
        model.eval()
        hidden = model.init_hidden(minibatch)
    total_loss = 0.
    nword = 0
    with torch.no_grad():
        for chunk, (input, target, sent_lens) in enumerate(dataloader):
            target_packed = pack_padded_sequence(target, sent_lens)[0]
            if not args.nnlm:
                output_me = memodel.forward(input, target, sent_lens)

            if not args.maxent:
                hidden = repackage_hidden(hidden)
                output_nn = model(input, target, hidden, sent_lens, ce=True)

            if args.nnlm is True:
                output = output_nn.view(-1, vocsize)
            elif args.maxent is True:
                output = output_me.view(-1, vocsize)
            else:
                output = torch.add(output_me.view(-1, vocsize), output_nn.view(-1, vocsize))

            loss = ce_crit(output, target_packed.view(-1))

            nvalidword = torch.sum(sent_lens).item()
            nword += nvalidword
            total_loss += loss.item()*nvalidword
    return total_loss / nword

#optimizer = torch.optim.LBFGS(model.parameters(), lr = lr, max_iter = 3, history_size = 5 )

def train():
    if not args.maxent:
        model.train()
        hidden = model.init_hidden(minibatch)
    total_loss = 0.
    total_nword = 0
    cur_loss = 0.
    nword = 0
    start_time = time.time()

    i = 0

    for chunk, (input, target, sent_lens) in enumerate(traindataloader):
        target_packed = pack_padded_sequence(target, sent_lens)[0]
        if not args.nnlm:
           output_me = memodel.forward(input, target, sent_lens)

        if not args.maxent:
           hidden = repackage_hidden(hidden)
           model.zero_grad()
           if args.noisedist == 'uniform':
             noise = noisesampler.draw_uniform(sent_lens[0].item(), args.ncesample)
           else:
             noise = noisesampler.draw(sent_lens[0].item(), args.ncesample)
             output_nn = model(input, target, hidden, sent_lens, noise=noise)
        
        if args.nnlm:
            if args.nce:
                loss = output_nn
            else:
                output = output_nn.view(-1, vocsize)
        elif args.maxent:
            output = output_me.view(-1, vocsize)
        else:
            output = torch.add(output_me.view(-1, vocsize), output_nn.view(-1, vocsize))

        if not args.nce:
            loss = ce_crit(output, target_packed.view(-1))

        # loss.backward()

        # if not args.nnlm:
        #     memodel.update(lr)
        # if not args.maxent:
        #     nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        #     #print('New period',i)
        #     i = i + 1
        #     for p in model.parameters():
        #         if p.grad is not None:
        #             #print(p.grad.data.size())
        #             p.data.add_(-lr, p.grad.data)
# pytorch LBFGS optimization
        
        optimizer = LBFGS_withAdam(model.parameters(), lr = lr, max_iter = 5, history_size = 10, use_Adam=use_Adam_flag)

        def closure():
            optimizer.zero_grad()
          
            if args.noisedist == 'uniform':
               noise = noisesampler.draw_uniform(sent_lens[0].item(), args.ncesample)
            else:
               noise = noisesampler.draw(sent_lens[0].item(), args.ncesample)
                     
            output_nn = model(input, target, hidden, sent_lens, noise=noise)
            
            if args.nnlm:
                if args.nce:
                    loss = output_nn
                else:
                    output = output_nn.view(-1, vocsize)
            elif args.maxent:
                output = output_me.view(-1, vocsize)
            else:
                output = torch.add(output_me.view(-1, vocsize), output_nn.view(-1, vocsize))

            
            loss = ce_crit(output, target_packed.view(-1))

            loss.backward(retain_graph = True)

            return loss
            
        if not args.nnlm:
            memodel.update(lr)
        if not args.maxent:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            loss = optimizer.step(closure)

        

        nvalidword = torch.sum(sent_lens).item()
        nword += nvalidword
        cur_loss += loss.item()*nvalidword
        total_nword += nvalidword
        total_loss += loss.item()*nvalidword

        if chunk % args.log_interval == 0 and chunk > 0:
            cur_loss = cur_loss / nword
            elapsed = time.time() - start_time
            # sys.stdout.write ('Epoch {:3d}   learn rate {:02.2f} train speed {:5.2f} word/sec, percent: {:5.2f} loss {:5.2f}, ppl {:8.2f} time: fw: {:.2f} s1: {:.2f} bw:{:.2f} \r'.format(epoch, lr, total_nword/elapsed, total_nword/TrainData.nword, cur_loss, math.exp(cur_loss), memodel.time_forward, memodel.time_step1-memodel.time_forward, memodel.time_backward))
            sys.stdout.write ('Epoch {:3d}   learn rate {:02.2f} train speed {:5.2f} word/sec, percent: {:5.2f} loss {:5.2f}, ppl {:8.2f} \r'.format(epoch, lr, total_nword/elapsed,  total_nword/TrainData.nword, cur_loss, math.exp(cur_loss) ))
            cur_loss = 0
            nword = 0
    total_loss = total_loss/total_nword
    elapsed = time.time() - start_time
    sys.stdout.write ('Epoch {:3d}   learn rate {:02.2f} speed {:5.2f} word/sec, train loss {:5.2f}, ppl {:8.2f},    '.format(epoch, lr, total_nword/elapsed, total_loss, math.exp(total_loss)))

try:
    if args.train:
        lr = args.learnrate
        best_val_loss = None
        for epoch in range(1, args.maxepoch+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(validdataloader)

            if not best_val_loss or val_loss < best_val_loss:
                if not args.maxent:
                    with open (args.save, 'wb') as f:
                        torch.save(model, f)
                best_val_loss = val_loss
            else:
                lr /= 4.0

            sys.stdout.write('valid loss {:5.2f}, ppl {:8.2f}\n'.format(val_loss, math.exp(val_loss)))

        if not args.maxent:
            with open (args.save, 'rb') as f:
                model = torch.load(f)
        test_loss = evaluate (testdataloader)
        print ('test loss {:5.2f}, ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))

    elif args.ppl:
        with open (args.load, 'rb') as f:
            model = torch.load(f)
        test_loss = calppl(testdataloader)
        print ('avglog loss {:5.2f}, PPL {:8.2f}'.format(test_loss, math.exp(test_loss)))

except KeyboardInterrupt:
    print ('Exiting from the training early')



