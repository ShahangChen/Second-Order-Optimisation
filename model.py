import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from device import device
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class RNNLM(nn.Module):
    def __init__ (self, emb_size, hidden_size, num_layer, minibatch, vocsize, droprate, noiseprob, nce=False, ncesample=1000, lognormconst=9.0, use_cell=True, use_rnn=False):
        super(RNNLM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layer = 1
        self.minibatch = minibatch
        self.vocsize = vocsize
        self.droprate = droprate
        self.use_cell = use_cell
        self.use_rnn = use_rnn
        self.nce = nce
        if self.nce is True:
            self.ce = False
        else:
            self.ce = True
        self.ncesample = ncesample
        self.lognormconst = lognormconst
        self.noiseprob = noiseprob

        self.dropout = nn.Dropout(self.droprate)
        self.emblayer = nn.Embedding(vocsize, emb_size)
        if (self.use_cell):
            if (self.use_rnn):
                self.rnnlayer = nn.RNNCell(emb_size, hidden_size)
            else:
                self.rnnlayer = nn.LSTMCell(emb_size, hidden_size)
        else:
            self.rnnlayer = nn.LSTM(emb_size, hidden_size, self.num_layer, dropout=self.droprate)
        self.outlayer = nn.Linear(hidden_size, self.vocsize)

        self.weight = nn.Parameter(torch.Tensor(self.vocsize, self.hidden_size))
        self.bias   = nn.Parameter(torch.Tensor(self.vocsize))
        self.init_weights()

        self.ce_crit = nn.CrossEntropyLoss(reduce=False)

    def init_weights(self):
        init_range = 0.1
        self.emblayer.weight.data.uniform_(-init_range, init_range)
        self.outlayer.weight.data.uniform_(-init_range, init_range)
        self.outlayer.bias.data.zero_()

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)



    def init_hidden(self, minibatch):
        weight = next(self.parameters()).data
        if (self.use_cell):
            hidden = weight.new_zeros([minibatch, self.hidden_size])
        else:
            hidden = (weight.new_zeros([1, minibatch, self.hidden_size]),
                       weight.new_zeros([1, minibatch, self.hidden_size]))
        return hidden

    def safe_log(self, tensor):
        EPSILON = 1e-10
        return torch.log(EPSILON+tensor)
        

    def forward(self, input, target, hidden, sent_lens, ce=False, noise=None):
        emb = self.emblayer(input)
        if(self.use_cell):
            if(self.use_rnn):
                hidden_output = torch.randn(emb.size(1), self.hidden_size).to(device)
                hidden_outputs = []
                for x in emb:
                    hidden_output = self.rnnlayer(x,  hidden)
                    hidden_outputs.append(hidden_output)
                hidden_outputs = pack_padded_sequence(torch.stack(hidden_outputs), sent_lens)
            else:
                hidden_output = torch.randn(emb.size(1), self.hidden_size).to(device)
                hidden_outputs = []
                for x in emb:
                    hidden_output, hidden = self.rnnlayer(x, (hidden_output, hidden))
                    hidden_outputs.append(hidden_output)
                hidden_outputs = pack_padded_sequence(torch.stack(hidden_outputs), sent_lens)
        else:
            emb = pack_padded_sequence(emb, sent_lens)
            hidden_outputs, hidden = self.rnnlayer(emb, hidden)

        ''' CE traing '''
        if self.ce is True or ce is True:
            output = F.linear(hidden_outputs[0], self.weight, self.bias)
            # output = self.outlayer(hidden_outputs[0])
        # ''' NCE training '''
        elif self.nce is True:
            ''' 
                target  size: seq_len, minibatch
                noise   size: seq_len, nsample
                indices size: seq_len, minibatch+nsample
                input   size: seq_len, minibatch, nhidden
            '''
            minibatch = target.size(-1)
            indices = torch.cat([target, noise], dim=-1)
            hidden_outputs = pad_packed_sequence(hidden_outputs)[0]
            hidden_outputs = hidden_outputs.contiguous()
            '''
                weight  size: seq_len, nhidden, minibatch+nsample
                bias    size: seq_len, 1,       minibatch+nsample
            '''
            weight  = self.weight.index_select(0, indices.view(-1)).view(*indices.size(), -1).transpose(1,2)
            bias    = self.bias.index_select(0, indices.view(-1)).view_as(indices).unsqueeze(1)
            '''
                out          size: seq_len, minibatch, minibatch+nsample
                target_score size: seq_len, minibatch, minibatch
                noise_score  size: seq_len, minibatch, nsample
            '''
            out = torch.baddbmm(1, bias, 1, hidden_outputs, weight)
            target_score, noise_score = out[:, :, :minibatch], out[:, :, minibatch:]
            target_score = target_score.sub(self.lognormconst).exp()
            noise_score  = noise_score.sub(self.lognormconst).exp()
            target_score = target_score.contiguous()
            noise_score = noise_score.contiguous()

            '''
                target_score      size: seq_len, minibatch
                target_noise_prob size: seq_len, minibatch
                noise_noise_prob   size: seq_len, minibatch, nsample
            '''
            index_slice = torch.arange(0, target_score.size(1)*target_score.size(2), target_score.size(1)).long()
            for i, v in enumerate(index_slice):
                index_slice[i] = index_slice[i]+i
            target_score = target_score.view(target_score.size(0), -1).contiguous()
            target_score = target_score[:, index_slice]
            ## target_score = target_score.view(target_score.size(0), -1)[:, index_slice]

            target_noise_prob = self.noiseprob[target.view(-1)].view_as(target_score)
            noise_noise_prob  = self.noiseprob[noise.view(-1)].view_as(noise).unsqueeze(1).expand_as(noise_score)

            model_loss = self.safe_log(target_score / (target_score+self.ncesample*target_noise_prob))
            noise_loss = torch.sum(self.safe_log((self.ncesample*noise_noise_prob) / (noise_score+self.ncesample*noise_noise_prob)), -1).squeeze()
            loss = -(model_loss+noise_loss)


            mask = input.gt(0.1)
            mask[0, :] = 1
            loss = torch.masked_select(loss, mask)
            return loss.mean()

        else:
            print ('need to be either ce or nce loss')
            exit()
        return output
        #return loss.mean()
        
