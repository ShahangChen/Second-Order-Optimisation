import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
#     hx, cx = hidden
#     gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
#     ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#     ingate     = F.sigmoid(ingate)
#     forgetgate = F.sigmoid(forgetgate)
#     cellgate   = F.tanh(cellgate)
#     outgate    = F.sigmoid(outgate)
#     cy = (forgetgate * cx) + (ingate * cellgate)
#     hy = outgate * F.tanh(cy)
#     return hy, cy

# def RnnCell():


# def LinearCell():


# def EmbeddingCell():


# def BackpropCell():


# class LSTMCell(RNNCellBase):
#     def __init__(self, input_size, hidden_size, bias=True):
#         super(LSTMCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
#         self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
#         if bias:
#             self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
#             self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
#         else:
#             self.register_parameter('bias_ih', None)
#             self.register_parameter('bias_hh', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, hx=None):
#         self.check_forward_input(input)
#         if hx is None:
#             hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
#             hx = (hx, hx)
#         self.check_forward_hidden(input, hx[0], '[0]')
#         self.check_forward_hidden(input, hx[1], '[1]')
#         return self._backend.LSTMCell(
#             input, hx,
#             self.weight_ih, self.weight_hh,
#             self.bias_ih, self.bias_hh,
#         )



import torch
from functools import reduce
from torch.optim.optimizer import Optimizer

class LBFGS_withAdam(Optimizer):
    """Implements L-BFGS algorithm.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(self, params, lr=3e-4, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None, use_Adam=False, Initial_flag=False):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn, use_Adam=use_Adam)
        super(LBFGS_withAdam, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        
    def __setstate__(self, state):
        super(LBFGS_withAdam, self).__setstate__(state)
        
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        use_Adam = group['use_Adam']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        # Added two key items for every state for later convenience in Adam --SH
        for p in self._params:
            state = self.state[p]
            state.setdefault('func_evals', 0)
            state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        abs_grad_sum = flat_grad.abs().sum()

        if abs_grad_sum <= tolerance_grad:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'ro' not in state:
                    state['ro'] = [None] * history_size
                    state['al'] = [None] * history_size
                ro = state['ro']
                al = state['al']

                for i in range(num_old):
                    ro[i] = 1. / old_dirs[i].dot(old_stps[i])

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(-al[i], old_dirs[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(al[i] - be_i, old_stps[i])

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone()
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / abs_grad_sum) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # optional line search: user function
            ls_func_evals = 0
            
            if use_Adam is True:
                # perform Adam Optimization - SH
                beta1, beta2=0.9, 0.999
                eps=1e-8
                
                offset = 0
            
                for p in self._params:
                    #print(p.size())
                    numel = p.numel()
                    grad = -1*(d[offset:offset + numel].view_as(p.data))
                    
                    #print("size of p",p.size(),"offset",offset,"&numel",numel,"size of grad", grad.size(),"size of d",d.size())
                    state = self.state[p]
                    # State initialization, originally state has n_iter & eval_func
                    if 'step' not in state:
                        #print("initialization implemented")
                        state.setdefault('step', 0) #add in another state
                        
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
    
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                   
                    denom = exp_avg_sq.sqrt().add_(eps)
    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = t * math.sqrt(bias_correction2) / bias_correction1
    
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    
                    offset += numel
                
                # Reevaluate the loss every step
                loss = float(closure())
                
                flat_grad = self._gather_flat_grad()
                
                abs_grad_sum = flat_grad.abs().sum()
                
                ls_func_evals = 1
                
                
            else:
                # no Adam, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    abs_grad_sum = flat_grad.abs().sum()
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            if abs_grad_sum <= tolerance_grad:
                break

            if gtd > -tolerance_change:
                break

            if d.mul(t).abs_().sum() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss