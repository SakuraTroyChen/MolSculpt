import time
import random
import numpy as np
import torch
import functools
# import dgl
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_dgl

def print_std(accs, stds, categories, append_mean=False):
    category_line = ' '.join(categories)
    if append_mean:
        category_line += ' Mean'

    line = ''
    if stds is None:
        for acc in accs:
            line += '{:0.1f} '.format(acc)
    else:
        for acc, std in zip(accs, stds):
            line += '{:0.1f}±{:0.1f} '.format(acc, std)

    if append_mean:
        line += '{:0.1f}'.format(sum(accs) / len(accs))
    print(category_line)
    print(line)


def simple_mixup(feat, y, mixup_alpha):
    B = feat.shape[0]
    device = feat.device
    list_f, list_y = feat, y
    permutation = torch.randperm(B)
    lam = np.random.beta(mixup_alpha, mixup_alpha, (B, 1))  # shape = [B,1]
    lam = torch.from_numpy(lam).to(device).float()
    f_ = (1-lam) * feat + lam * feat[permutation]
    y_ = (1-lam) * y + lam * y[permutation]
    list_f = torch.cat((list_f, f_), dim=0)  # shape = [B, D]
    list_y = torch.cat((list_y, y_), dim=0)  # shape = [B, C]
    return list_f, list_y


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert n > 0
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def timeit(func):
    @functools.wraps(func)
    def timed(*args, **kw):
        ts = time.time()
        print_time_info('Method: %s started!' % (func.__name__), dash_top=True)
        result = func(*args, **kw)
        te = time.time()
        print_time_info('Method: %s cost %.2f sec!' %
                        (func.__name__, te-ts), dash_bot=True)
        return result
    return timed


def print_time_info(string, end='\n', dash_top=False, dash_bot=False, file=None):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(time.time())))
    string = "[%s] %s" % (times, str(string))
    if dash_top:
        print(len(string) * '-', file=file)
    print(string, end=end, file=file)
    if dash_bot:
        print(len(string) * '-', file=file)


def worker_seed_init(idx, seed):
    torch_seed = torch.initial_seed()
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    seed = idx + seed + torch_seed
    random.seed(seed)
    np.random.seed(seed)

# SEED
def set_seed(seed, device=None):
    if_cuda = torch.cuda.is_available()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # dgl.seed(seed)


def write_log(print_str, log_file, print_=True):
    if print_:
        print_time_info(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


def sinkhorn(s: Tensor, nrows: Tensor = None, ncols: Tensor = None,
             unmatchrows: Tensor = None, unmatchcols: Tensor = None,
             dummy_row: bool = False, max_iter: int = 10, tau: float = 1., batched_operation: bool = False) -> Tensor:
    """
    Pytorch implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.transpose(1, 2)
        nrows, ncols = ncols, nrows
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True
    
    if nrows is None:
        nrows = torch.tensor([s.shape[1] for _ in range(batch_size)], device=s.device)
    if ncols is None:
        ncols = torch.tensor([s.shape[2] for _ in range(batch_size)], device=s.device)

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if torch.any(transposed_batch):
        s_t = s.transpose(1, 2)
        s_t = torch.cat((
            s_t[:, :s.shape[1], :],
            torch.full((batch_size, s.shape[1], s.shape[2] - s.shape[1]), -float('inf'), device=s.device)), dim=2)
        s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, s)

        new_nrows = torch.where(transposed_batch, ncols, nrows)
        new_ncols = torch.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

        
    # operations are performed on log_s
    log_s = s / tau

    if False:
        row_mask = torch.zeros(batch_size, log_s.shape[1], 1, dtype=torch.bool, device=log_s.device)
        col_mask = torch.zeros(batch_size, 1, log_s.shape[2], dtype=torch.bool, device=log_s.device)
        for b in range(batch_size):
            row_mask[b, :nrows[b], 0] = 1
            col_mask[b, 0, :ncols[b]] = 1
    else:
        row_mask = torch.arange(log_s.shape[1], device=log_s.device).view(1, -1) < nrows.view(-1, 1)
        col_mask = torch.arange(log_s.shape[2], device=log_s.device).view(1, -1) < ncols.view(-1, 1)
        row_mask = row_mask.unsqueeze(2)
        col_mask = col_mask.unsqueeze(1)
    
    if batched_operation:
        if False:
            for b in range(batch_size):
                log_s[b, nrows[b]:, :] = -float('inf')
                log_s[b, :, ncols[b]:] = -float('inf')
        else:
            # log_s.masked_fill_(~row_mask, -float('inf'))
            # log_s.masked_fill_(~col_mask, -float('inf'))
            min_float = torch.finfo(torch.float).min
            log_s.masked_fill_(~row_mask, min_float)
            log_s.masked_fill_(~col_mask, min_float)

        zero_like = torch.zeros_like(log_s).detach()
        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                log_s = log_s - torch.where(row_mask, log_sum, zero_like)
                # assert not torch.any(torch.isnan(log_s))
            else:
                log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                log_s = log_s - torch.where(col_mask, log_sum, zero_like)
                # assert not torch.any(torch.isnan(log_s))

        ret_log_s = log_s
    else:
        ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=log_s.device,
                               dtype=log_s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                    log_s_b = log_s_b - torch.where(row_mask_b, log_sum, torch.zeros_like(log_sum))
                else:
                    log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                    log_s_b = log_s_b - torch.where(col_mask_b, log_sum, torch.zeros_like(log_sum))

            ret_log_s[b, row_slice, col_slice] = log_s_b

    if torch.any(transposed_batch):
        s_t = ret_log_s.transpose(1, 2)
        s_t = torch.cat((
            s_t[:, :ret_log_s.shape[1], :],
            torch.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2] - ret_log_s.shape[1]), -float('inf'),
                       device=log_s.device)), dim=2)
        ret_log_s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, ret_log_s)

    if transposed:
        ret_log_s = ret_log_s.transpose(1, 2)

    return torch.exp(ret_log_s)


def to_dgl_batch(batch):
    assert isinstance(batch, Batch)
    graph = to_dgl(batch)
    graph.set_batch_num_nodes(batch.ptr[1:] - batch.ptr[:-1])
    graph.ptr = batch.ptr
    if hasattr(batch, 'max_num_nodes'):
        graph.max_num_nodes = batch.max_num_nodes
    return graph
