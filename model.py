import nn
import utils

import torch
from torch.nn import functional as F
from torch.nn import init

import numpy as np

import librosa
import math
import scipy
from scipy.signal import lfilter
#import matplotlib.pyplot as plt
from spectrum import poly2lsf, lsf2poly
import pyworld as pw


class SampleRNN(torch.nn.Module):

    def __init__(self, frame_sizes, n_rnn, dim, learn_h0, q_levels, M,
                 weight_norm):
        super().__init__()

        self.dim = dim
        self.q_levels = q_levels
        self.M = M

        ns_frame_samples = map(int, np.cumprod(frame_sizes))
        self.frame_level_rnns = torch.nn.ModuleList([
            FrameLevelRNN(
                frame_size, n_frame_samples, n_rnn, dim, M, learn_h0, weight_norm
            )
            for (frame_size, n_frame_samples) in zip(
                frame_sizes, ns_frame_samples
            )
        ])

        self.sample_level_mlp = SampleLevelMLP(
            frame_sizes[0], dim, q_levels, weight_norm
        )

    @property
    def lookback(self):
        return self.frame_level_rnns[-1].n_frame_samples


class FrameLevelRNN(torch.nn.Module):

    def __init__(self, frame_size, n_frame_samples, n_rnn, dim, M,
                 learn_h0, weight_norm):
        super().__init__()

        self.frame_size = frame_size
        self.n_frame_samples = n_frame_samples
        self.dim = dim
        self.M = M

        h0 = torch.zeros(n_rnn, dim)
        if learn_h0:
            self.h0 = torch.nn.Parameter(h0)
        else:
            #self.register_buffer('h0', torch.autograd.Variable(h0))
            with torch.no_grad: # according to user warning (BGF 2020)
                self.register_buffer('h0', h0)

        self.input_expand = torch.nn.Conv1d(
            in_channels=n_frame_samples,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform_(self.input_expand.weight) # according to user warning (BGF 2020)
        #init.kaiming_uniform(self.input_expand.weight) 
        init.constant_(self.input_expand.bias, 0) # according to user warning (BGF 2020)
        #init.constant(self.input_expand.bias, 
        if weight_norm:
            self.input_expand = torch.nn.utils.weight_norm(self.input_expand)


        #################
        self.cond_expand = torch.nn.Conv1d(
            in_channels=M,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform_(self.cond_expand.weight) # according to user warning (BGF 2020)
        #init.kaiming_uniform(self.cond_expand.weight) 
        init.constant_(self.cond_expand.bias, 0) # according to user warning (BGF 2020)
        #init.constant(self.cond_expand.bias, 
        if weight_norm:
            self.cond_expand = torch.nn.utils.weight_norm(self.cond_expand)
        ################

        self.rnn = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=n_rnn,
            batch_first=True
        )
        for i in range(n_rnn):
            nn.concat_init(
                getattr(self.rnn, 'weight_ih_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, nn.lecun_uniform]
            )
            init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), 0)

            nn.concat_init(
                getattr(self.rnn, 'weight_hh_l{}'.format(i)),
                [nn.lecun_uniform, nn.lecun_uniform, init.orthogonal_]
            )
            init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), 0)

        self.upsampling = nn.LearnedUpsampling1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=frame_size
        )

        init.uniform_(
            self.upsampling.conv_t.weight, -np.sqrt(6 / dim), np.sqrt(6 / dim)
        )
        init.constant_(self.upsampling.bias, 0)
        if weight_norm:
            self.upsampling.conv_t = torch.nn.utils.weight_norm(
                self.upsampling.conv_t
            )

    def forward(self, prev_samples, hf, upper_tier_conditioning, hidden):
        (batch_size, _, _) = prev_samples.size()


        input = self.input_expand(
          prev_samples.permute(0, 2, 1)
        ).permute(0, 2, 1)

        ##################
        input_cond = self.cond_expand(
          hf.permute(0, 2, 1)
        ).permute(0, 2, 1)
        ##################

        if upper_tier_conditioning is not None:
            input += upper_tier_conditioning
        
        input += input_cond

        reset = hidden is None

        if hidden is None:
            (n_rnn, _) = self.h0.size()
            hidden = self.h0.unsqueeze(1) \
                            .expand(n_rnn, batch_size, self.dim) \
                            .contiguous()

        (output, hidden) = self.rnn(input, hidden)

        output = self.upsampling(
            output.permute(0, 2, 1)
        ).permute(0, 2, 1)

        return (output, hidden)


class SampleLevelMLP(torch.nn.Module):

    def __init__(self, frame_size, dim, q_levels, weight_norm):
        super().__init__()

        self.q_levels = q_levels

        self.embedding = torch.nn.Embedding(
            self.q_levels,
            self.q_levels
        )

        self.input = torch.nn.Conv1d(
            in_channels=q_levels,
            out_channels=dim,
            kernel_size=frame_size,
            bias=False
        )
        init.kaiming_uniform_(self.input.weight)
        if weight_norm:
            self.input = torch.nn.utils.weight_norm(self.input)

        self.hidden = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )
        init.kaiming_uniform_(self.hidden.weight)
        init.constant_(self.hidden.bias, 0)
        if weight_norm:
            self.hidden = torch.nn.utils.weight_norm(self.hidden)

        self.output = torch.nn.Conv1d(
            in_channels=dim,
            out_channels=q_levels,
            kernel_size=1
        )
        nn.lecun_uniform(self.output.weight)
        init.constant_(self.output.bias, 0)
        if weight_norm:
            self.output = torch.nn.utils.weight_norm(self.output)

    def forward(self, prev_samples, upper_tier_conditioning):
        (batch_size, _, _) = upper_tier_conditioning.size()

        prev_samples = self.embedding(
            prev_samples.contiguous().view(-1)
        ).view(
            batch_size, -1, self.q_levels
        )

        prev_samples = prev_samples.permute(0, 2, 1)
        upper_tier_conditioning = upper_tier_conditioning.permute(0, 2, 1)

        x = F.relu(self.input(prev_samples) + upper_tier_conditioning)
        x = F.relu(self.hidden(x))
        x = self.output(x).permute(0, 2, 1).contiguous()

        return F.log_softmax(x.view(-1, self.q_levels), dim=1) \
                .view(batch_size, -1, self.q_levels) ##### VÉRIFIER dim ##### IMPORTANT !!!


class Runner:

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden_states = {rnn: None for rnn in self.model.frame_level_rnns}

    def run_rnn(self, rnn, prev_samples, hf, upper_tier_conditioning):
        (output, new_hidden) = rnn(
            prev_samples, hf, upper_tier_conditioning, self.hidden_states[rnn]
        )
        self.hidden_states[rnn] = new_hidden.detach()
        return output


class Predictor(Runner, torch.nn.Module):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, input_sequences, reset):
        if reset:
            self.reset_hidden_states()

        (batch_size, _) = input_sequences.size()
        upper_tier_conditioning = None
        for rnn in reversed(self.model.frame_level_rnns):
            from_index = self.model.lookback - rnn.n_frame_samples
            to_index = -rnn.n_frame_samples + 1
            prev_samples = 2 * utils.linear_dequantize(
                input_sequences[:, from_index : to_index],
                self.model.q_levels
            )
            prev_samples = prev_samples.contiguous().view(
                batch_size, -1, rnn.n_frame_samples
            )

            if upper_tier_conditioning == None:
                [N, C, L] = prev_samples.size()
                hf = torch.zeros([N, C, self.model.M]) # + 1 (for the rms nrj)

                for batch_ind in range(N):
                    for frame_ind in range(C):
                        temp_seq = prev_samples[batch_ind, frame_ind, :].cpu().numpy()

                        # Calculs lsf
                        # Calcul du bruit blanc gaussien à ajouter au signal pour éviter "ill-conditioned" matrix
                        # rms = np.sqrt(np.mean(temp_seq**2))
                        # print(rms)
                        # n_next = 1
                        # if rms == 0:
                        #     while rms == 0:
                        #         temp_seq = prev_samples[batch_ind, frame_ind+n_next, :].cpu().numpy()
                        #         rms = np.sqrt(np.mean(temp_seq**2))
                        #         n_next += 1
                        #         print("flag")

                        rms = 0.1
                        var = rms * 0.0001 # (-40db)
                        std = math.sqrt(var)
                        mu, sigma = 0, std # mean = 0 and standard deviation

                        temp_seq = temp_seq + np.random.normal(mu, sigma, L)
                    
                        # hanning windowing
                        temp_seq = temp_seq*np.hanning(L)

                        a = librosa.core.lpc(temp_seq, self.model.M-1)
                        lsf = torch.from_numpy(np.asarray(poly2lsf(a)))

                        # Calculs nrj
                        residu = lfilter(a, 1, temp_seq)
                        nrjRMS_residu = torch.from_numpy(np.asarray(np.sqrt(np.mean(residu**2))))

                        # print(lsf)
                        # print(nrjRMS_residu)

                        hf[batch_ind, frame_ind, :-1] = lsf
                        hf[batch_ind, frame_ind, -1] = nrjRMS_residu
                        # print(hf[0, 0, :])
                        # exit()

                hf = hf.cuda()
                # import sys
                # torch.set_printoptions(threshold=sys.maxsize)
                # print(hf)
                # exit()

            
            hf_out = utils.tile(hf, 1, int(L/rnn.n_frame_samples))

            upper_tier_conditioning = self.run_rnn(
                rnn, prev_samples, hf_out, upper_tier_conditioning
            )

        bottom_frame_size = self.model.frame_level_rnns[0].frame_size
        mlp_input_sequences = input_sequences \
            [:, self.model.lookback - bottom_frame_size :]

        return self.model.sample_level_mlp(
            mlp_input_sequences, upper_tier_conditioning
        )


class Generator(Runner):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.cuda = cuda

    def __call__(self, n_seqs, seq_len):
        # generation doesn't work with CUDNN for some reason
        torch.backends.cudnn.enabled = False
        print("Me generating")
        self.reset_hidden_states()

        bottom_frame_size = self.model.frame_level_rnns[0].n_frame_samples
        sequences = torch.LongTensor(n_seqs, self.model.lookback + seq_len) \
                         .fill_(utils.q_zero(self.model.q_levels))
        frame_level_outputs = [None for _ in self.model.frame_level_rnns]

        for i in range(self.model.lookback, self.model.lookback + seq_len):
            for (tier_index, rnn) in \
                    reversed(list(enumerate(self.model.frame_level_rnns))):
                if i % rnn.n_frame_samples != 0:
                    continue

                # prev_samples = torch.autograd.Variable(
                #     2 * utils.linear_dequantize(
                #         sequences[:, i - rnn.n_frame_samples : i],
                #         self.model.q_levels
                #     ).unsqueeze(1),
                #     volatile=True
                # )
                with torch.no_grad():
                    prev_samples = (2 * utils.linear_dequantize(sequences[:, i - rnn.n_frame_samples : i], self.model.q_levels ).unsqueeze(1))


                    [_,ind_,_] = prev_samples.size()

                    hf_tensor = torch.zeros((n_seqs, ind_, self.model.M)).float()
                    for s in range(n_seqs):
                        for frm in range(ind_):
                            if s == 0:
                                hf_tensor[s, frm, :] = torch.tensor([0.202038314729498,	0.259133820904786,	0.650005536799759,	0.774293997015348,	1.15643122931357,\
                                	                            1.40107779574558,	1.67377481530053,	2.00228013396680,	2.37544570740133,	2.85229620658522, 0.00339120439437788], dtype=torch.float64)
                            elif s == 1:
                                hf_tensor[s, frm, :] = torch.tensor([0.202038314729498,	0.259133820904786,	0.650005536799759,	0.774293997015348,	1.15643122931357,\
                                	                            1.40107779574558,	1.67377481530053,	2.00228013396680,	2.37544570740133,	2.85229620658522, 0.0339120439437788], dtype=torch.float64)
                            elif s == 2:
                                hf_tensor[s, frm, :] = torch.tensor([0.202038314729498,	0.259133820904786,	0.650005536799759,	0.774293997015348,	1.15643122931357,\
                                	                            1.40107779574558,	1.67377481530053,	2.00228013396680,	2.37544570740133,	2.85229620658522, 0.339120439437788], dtype=torch.float64)
  
  
                if self.cuda:
                    prev_samples = prev_samples.cuda()
                    hf_tensor = hf_tensor.cuda()

                if tier_index == len(self.model.frame_level_rnns) - 1:
                    upper_tier_conditioning = None
                else:
                    frame_index = (i // rnn.n_frame_samples) % \
                        self.model.frame_level_rnns[tier_index + 1].frame_size
                    upper_tier_conditioning = \
                        frame_level_outputs[tier_index + 1][:, frame_index, :] \
                                           .unsqueeze(1)

                frame_level_outputs[tier_index] = self.run_rnn(
                    rnn, prev_samples,hf_tensor, upper_tier_conditioning
                )

            # prev_samples = torch.autograd.Variable(
            #     sequences[:, i - bottom_frame_size : i],
            #     volatile=True
            # )

            with torch.no_grad():
                prev_samples = sequences[:, i - bottom_frame_size : i]

            if self.cuda:
                prev_samples = prev_samples.cuda()
            upper_tier_conditioning = \
                frame_level_outputs[0][:, i % bottom_frame_size, :] \
                                      .unsqueeze(1)
            sample_dist = self.model.sample_level_mlp(
                prev_samples, upper_tier_conditioning
            ).squeeze(1).exp_().data
            sequences[:, i] = sample_dist.multinomial(1).squeeze(1)

        torch.backends.cudnn.enabled = True

        return sequences[:, self.model.lookback :]
