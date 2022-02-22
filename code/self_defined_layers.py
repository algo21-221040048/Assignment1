# This part is relevant with table 7 in this paper
import torch
import numpy as np
import torch.nn.functional as f
from audtorch.metrics.functional import pearsonr


KERNEL_LENGTH_EXTRACT_LAYER = 10
STRIDE_LENGTH_EXTRACT_LAYER = 10
KERNEL_LENGTH_POOLING_LAYER = 3
STRIDE_LENGTH_POOLING_LAYER = 3


def ts_corr_single(x, kernel_length, stride_length):  # this function has already flattened
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = torch.cat(
        [f.unfold(x.float(), kernel_size=(2, kernel_length), stride=(1, stride_length), dilation=(i, 1)) for i in
         range(1, x.size()[2])], 2)
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)  # get all possible c_n^2 cases
    result = torch.cat(
        [get_corr(medium[i][:kernel_length], medium[i][kernel_length:]).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_corr(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    rows = rows * (rows - 1) / 2
    result = torch.cat([ts_corr_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def get_corr(X, Y):  # 计算两个向量pearsonr相关系数, note Bessel's correction and False means not to use it
    # corr = torch.sum((x1-x1.mean()) * (x2-x2.mean())) / (x2.std(unbiased=False) * x1.std(unbiased=False) * len(x1))
    corr = pearsonr(X, Y)
    return corr


def ts_cov_single(x, kernel_length, stride_length):  # this function has already flattened
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = torch.cat(
        [f.unfold(x.float(), kernel_size=(2, kernel_length), stride=(1, stride_length), dilation=(i, 1)) for i in
         range(1, x.size()[2])], 2)
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)  # get all possible c_n^2 cases
    result = torch.cat(
        [get_cov(medium[i][:kernel_length], medium[i][kernel_length:]).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_cov(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    rows = rows * (rows - 1) / 2
    result = torch.cat([ts_cov_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def get_cov(X, Y):  # 不考虑修正
    assert len(X) == len(Y)
    cov = torch.sum((X-X.mean()) * (Y-Y.mean())) / len(Y)
    return cov


def ts_stddev_single(x, kernel_length, stride_length):
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [(medium[i].std(unbiased=False)).unsqueeze(0).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_stddev(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_stddev_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def ts_zscore_single(x, kernel_length, stride_length):
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [(medium[i].mean() / medium[i].std(unbiased=False)).unsqueeze(0).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_zscore(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_zscore_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def ts_return_single(x, kernel_length, stride_length):
    assert kernel_length == stride_length
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [get_return(medium[i][0], medium[i][-1]).unsqueeze(0).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_return(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_return_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def get_return(X, Y):
    return (Y - X)/X - 1


def ts_decaylinear_single(x, kernel_length, stride_length):
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [torch.from_numpy(np.average(medium[i], weights=range(1, kernel_length+1, 1)).reshape(-1)).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result.float()


def ts_decaylinear(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_decaylinear_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def ts_min_single(x, kernel_length, stride_length):
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [torch.min(medium[i]).unsqueeze(0).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_min(x, kernel_length=KERNEL_LENGTH_POOLING_LAYER, stride_length=STRIDE_LENGTH_POOLING_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_min_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def ts_max_single(x, kernel_length, stride_length):
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [torch.max(medium[i]).unsqueeze(0).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_max(x, kernel_length=KERNEL_LENGTH_POOLING_LAYER, stride_length=STRIDE_LENGTH_POOLING_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_max_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def ts_sum_single(x, kernel_length, stride_length):
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [torch.sum(medium[i]).unsqueeze(0).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_sum(x, kernel_length=KERNEL_LENGTH_POOLING_LAYER, stride_length=STRIDE_LENGTH_POOLING_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_sum_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def ts_mean_single(x, kernel_length, stride_length):
    assert len(x.size()) == 4
    # assert x.dtype == torch.float64
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    B, W, L = medium.size()
    medium = medium.permute(0, 2, 1)
    medium = medium.squeeze(0)
    result = torch.cat(
        [torch.mean(medium[i]).unsqueeze(0).unsqueeze(0) for i in
         range(medium.size()[0])], 0)
    assert result.size()[0] == medium.size()[0]
    return result


def ts_mean_pool(x, kernel_length=KERNEL_LENGTH_POOLING_LAYER, stride_length=STRIDE_LENGTH_POOLING_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_mean_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)


def ts_mean_extract(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert len(x.size()) == 4
    rows = x.size()[-2]
    result = torch.cat([ts_mean_single(x[i].unsqueeze(0), kernel_length, stride_length) for i in range(x.size()[0])], 0)
    return result.view(x.size()[0], 1, int(rows), -1)






