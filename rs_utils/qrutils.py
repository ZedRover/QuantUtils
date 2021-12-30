import numpy as np
import pandas as pd
import scipy
from numba import jit
import datetime
import warnings
from matplotlib import pyplot as plt
from tqdm import tqdm, tqdm_notebook
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from math import erf

# from .verbose_utils.deprecate_verbose import deprecate_function

# -------------------------------------------for structure----------------------------
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# --------------------------------------for debug info---------------------------------
def getProcessMemoryUsage():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MB


# --------------------------------------for math ops--------------------------------------

@jit()
def _cor_calc(x, y):
    n = len(x)
    return np.nansum((x - np.nanmean(x)) * (y - np.nanmean(y))) / np.sqrt(np.nanvar(x) * np.nanvar(y)) / n


@jit()
def _cor_calc_mat(X, Y):
    n = len(X)
    X_std = np.std(X, axis=0).reshape(-1, 1)
    Y_std = np.std(Y, axis=0).reshape(-1, 1)
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    return np.dot((X - X_mean).T, (Y - Y_mean)) / np.dot(X_std, Y_std.T) / n


def corr(x, y, mask=None, x_percent=None, y_percent=None):
    """
    :x: list-like. ndim=1/2. x_ndim and y_ndim cannot both be 2
    :y: list-like
    :mask: np.mask, 1 blocked, 0 exposed
    :x_percent: two element list, 0-100. If both x_percent and y_percent are not None, only calculate x_percent
    :y_percent: two element list, 0-100

    return: corr array
    """
    # x_percenty_percent: from 0-100

    if hasattr(x, 'values'):
        x = x.values
    if hasattr(y, 'values'):
        y = y.values

    if x.shape[0] != y.shape[0]:
        raise Warning('the shape of x and y is not consistent!')
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if y.ndim == 2 and 1 in y.shape:
        y = y.reshape(-1)

    if y.ndim == 1 and x.ndim != 1:
        temp = y
        y = x
        x = temp

    if x.ndim == 1 and y.ndim == 1:
        if x_percent is not None:
            mask = np.ones_like(x)
            mask[np.where((x < np.percentile(x, x_percent[0])) | (x > np.percentile(x, x_percent[1])))] = 0
        elif y_percent is not None:
            mask = np.ones_like(y)
            mask[np.where((y < np.percentile(y, y_percent[0])) | (y > np.percentile(y, y_percent[1])))] = 0

    # nan
    temp_df = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1, sort=False).replace([np.inf, -np.inf], np.nan)
    if temp_df.isnull().any(axis=None):
        temp_df = temp_df.dropna()
        if len(x.shape)==1:
            x_shape = 1
        else:
            x_shape = x.shape[1]
        x = temp_df.iloc[:, :x_shape].values
        y = temp_df.iloc[:, x_shape:].values

    if mask is not None:
        x = x[mask == 0]
        y = y[mask == 0]

    if (x.ndim == 1 or 1 in x.shape) and (y.ndim == 1 or 1 in y.shape):
        return _cor_calc(x, y)
    else:
        if y.ndim == 1:
            y = y.reshape(-1,1)
        if x.ndim == 1:
            x = x.reshape(-1,1)
        res = _cor_calc_mat(x, y)
        if 1 in res.shape:
            res = res.reshape(-1)
        return res


def internally_studentized_residual(x, y):
    """
    residual>=3 indicates strong outlier suspection
    residual>=2 indicates outlier suspection
    """
    x = np.array(x)
    y = np.array(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    n = len(x)
    diff_mean_sqr = np.dot((x - mean_x), (x - mean_x))
    beta1 = np.dot((x - mean_x), (y - mean_y)) / diff_mean_sqr
    beta0 = mean_y - beta1 * mean_x
    y_hat = beta0 + beta1 * x
    residuals = y - y_hat
    h_ii = (x - mean_x) ** 2 / diff_mean_sqr + (1 / n)
    var_e = np.sqrt(sum((y - y_hat) ** 2) / (n - 2))
    SE_regression = var_e * ((1 - h_ii) ** 0.5)
    studentized_residuals = residuals / SE_regression
    return studentized_residuals


def corr_deoutlier(x, y, verbose=True):
    sr1 = internally_studentized_residual(x, y)
    sr2 = internally_studentized_residual(y, x)
    sr_min = np.array([min(abs(r1), abs(r2)) for r1, r2 in zip(sr1, sr2)])
    bad_point = np.where(sr_min > 3)[0]
    if verbose:
        print(f'delete {len(bad_point)} outlier')
    x_m = np.delete(np.array(x), bad_point, axis=0)
    y_m = np.delete(np.array(y), bad_point, axis=0)
    return corr(x_m, y_m)


# --------------------------------------for math ops--------------------------------------

@jit(nogil=True, nopython=True)
def _time_diff(timestamps, a, diff_time, larger=True):
    assert (a.shape[0] == timestamps.shape[0])

    result_len = a.shape[0]

    if abs(diff_time) < 1e4:
        result = np.full(result_len, np.nan)
        if diff_time > 0:
            for i in range(diff_time, result_len):
                if timestamps[i] - timestamps[i - diff_time] > 60 * 60 * 10 ** 9:
                    result[i] = np.nan
                else:
                    result[i] = a[i] - a[i - diff_time]
        elif diff_time < 0:
            for i in range(result_len + diff_time):
                if timestamps[i - diff_time] - timestamps[i] > 60 * 60 * 10 ** 9:
                    result[i] = np.nan
                else:
                    result[i] = a[i] - a[i - diff_time]
        else:
            result = np.zeros_like(a, dtype=np.float64)

    else:
        result = np.zeros_like(a, dtype=np.float64)
        if larger:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] >= diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10 ** 9:
                        result[i] = a[i] - a[j]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while j < result_len and timestamps[j] - timestamps[i] < -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10 ** 9:
                            result[i] = np.nan
                        else:
                            result[i] = a[i] - a[j]
                    else:
                        result[i] = np.nan

            else:
                pass
        else:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] > diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10 ** 9:
                        result[i] = a[i] - a[j + 1]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while j < result_len and timestamps[j] - timestamps[i] <= -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10 ** 9:
                            result[i] = np.nan
                        else:
                            result[i] = a[i] - a[j - 1]
                    else:
                        result[i] = np.nan

            else:
                pass

    return result


def time_diff(timestamps, a, diff_time):
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.values
    if hasattr(a, 'values'):
        a = a.values
    diff_time = int(diff_time)

    return _time_diff(timestamps, a, diff_time)


def log_return(timestamps, a, diff_time, shift_time=0):
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.values
    if hasattr(a, 'values'):
        a = a.values

    return _log_return(timestamps, a, diff_time, shift_time)


@jit(nogil=True)
def _log_return(timestamps, a, diff_time, shift_time=0, larger=True, error_pct=0.2):
    assert (a.shape[0] == timestamps.shape[0])
    result_len = a.shape[0]
    result = np.zeros_like(a, dtype=np.float64)
    index_j = np.zeros_like(a, dtype=int)
    ts_j = np.zeros_like(a, dtype=int)
    if diff_time == 0:
        return result
    if larger:
        j = 0
        shift_i = 0
        for i in range(result_len):
            if diff_time > 0:
                while timestamps[i] - timestamps[j + 1] >= diff_time or timestamps[i] - timestamps[j] >= diff_time * (
                        1 + error_pct):
                    j += 1
            else:
                while j + 1 < result_len and timestamps[j] - timestamps[i] < -diff_time:
                    j += 1
                interval1 = timestamps[j] - timestamps[i] + diff_time
                interval2 = timestamps[i] - diff_time - timestamps[j - 1]
                # when diff time is relatively small, control the interval precision
                if j > 0 and \
                        (abs(interval1 - interval2) < 0.6 * 10 ** 9 or (
                                timestamps[j] - timestamps[i]) > 60 * 60 * 10 ** 9) and \
                        interval1 > interval2:
                    j = j - 1
            if shift_time > 0:
                while timestamps[i] - timestamps[shift_i] > shift_time * (1 + error_pct):
                    shift_i += 1
            elif shift_time < 0:
                while shift_i + 1 < result_len and (shift_i == i or timestamps[shift_i] - timestamps[i] <= -shift_time):
                    shift_i += 1
                # when shift_time is relatively small by default, control the interval precision
                interval1 = timestamps[shift_i] - timestamps[i] + shift_time
                interval2 = timestamps[i] - shift_time - timestamps[
                    shift_i - 1] if shift_i > 0 and shift_i - 1 != i else 60 * 60 * 10 ** 9
                shift_i = shift_i - 1 if interval1 > interval2 else shift_i
            else:
                shift_i = i
            if 0 <= j < result_len and abs(timestamps[i] - timestamps[j]) < 60 * 60 * 10 ** 9:
                result[i] = np.log(a[shift_i] / a[j])
            else:
                result[i] = np.nan

            index_j[i] = j
            ts_j[i] = timestamps[shift_i]
    else:
        j = 0 if diff_time > 0 else -1
        shift_i = 0 if shift_time > 0 else -1
        for i in range(result_len):
            if diff_time > 0:
                while timestamps[i] - timestamps[j] > diff_time:
                    j += 1
            else:
                while j < result_len and timestamps[j + 1] - timestamps[i] <= -diff_time:
                    j += 1
            if shift_time > 0:
                while timestamps[i] - timestamps[shift_i] > shift_time * (1 + error_pct):
                    shift_i += 1
            elif shift_time < 0:
                while shift_i + 1 < result_len and timestamps[shift_i + 1] - timestamps[i] <= -shift_time * (
                        1 + error_pct):
                    shift_i += 1
            else:
                shift_i = i
            if result_len > j >= 0 and abs(timestamps[j] - timestamps[i]) < 60 * 60 * 10 ** 9:
                result[i] = np.log(a[shift_i] / a[j])
            else:
                result[i] = np.nan

    return result


@jit(nogil=True, nopython=True)
def _time_shift(timestamps, a, diff_time, larger=True):
    assert (a.shape[0] == timestamps.shape[0])

    result_len = a.shape[0]

    if abs(diff_time) < 1e4:
        result = np.full(result_len, np.nan)
        if diff_time > 0:
            for i in range(diff_time, result_len):
                if timestamps[i] - timestamps[i - diff_time] > 60 * 60 * 10 ** 9:
                    result[i] = np.nan
                else:
                    result[i] = a[i - diff_time]
        elif diff_time < 0:
            for i in range(result_len + diff_time):
                if timestamps[i - diff_time] - timestamps[i] > 60 * 60 * 10 ** 9:
                    result[i] = np.nan
                else:
                    result[i] = a[i - diff_time]
        else:
            result = a

    else:
        result = np.zeros_like(a, dtype=np.float64)
        if larger:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] >= diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10 ** 9:
                        result[i] = a[j]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while j < result_len and timestamps[j] - timestamps[i] < -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10 ** 9:
                            result[i] = np.nan
                        else:
                            result[i] = a[j]
                    else:
                        result[i] = np.nan

            else:
                pass
        else:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] > diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10 ** 9:
                        result[i] = a[j + 1]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while j < result_len and timestamps[j] - timestamps[i] <= -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10 ** 9:
                            result[i] = np.nan
                        else:
                            result[i] = a[j - 1]
                    else:
                        result[i] = np.nan

            else:
                pass

    return result


def time_shift(timestamps, a, diff_time):
    """
    :param timestamps:
    :param a:
    :param diff_time: time diff in nanos
    :return:
    """
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.values
    if hasattr(a, 'values'):
        a = a.values
    diff_time = int(diff_time)

    return _time_shift(timestamps, a, diff_time)


@jit(nogil=True)
def ewm_sum_(timestamps, a, half_life):
    lambda_ = np.log(2) / half_life
    result = np.zeros_like(a)
    result_len = result.shape[0]
    result[0] = a[0]
    for i in range(1, result_len):
        timediff = timestamps[i] - timestamps[i - 1]
        old_weight = np.exp(-timediff * lambda_)
        result[i] = result[i - 1] * old_weight + a[i]
    return result


def ewm_sum(timestamps, a, half_life):
    """
    :param timestamps:
    :param a:
    :param half_life: half decay in nanos
    :return:
    """
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.values
    if hasattr(a, 'values'):
        a = a.values
    return ewm_sum_(timestamps, a, half_life)


def todatetime(timestamps):
    return pd.to_datetime(timestamps) + datetime.timedelta(hours=8)


def time_rolling(df, time_idx, window, min_periods=1):
    df = df.copy()
    if isinstance(time_idx, str):
        time_idx = df[time_idx]
    df.index = pd.to_datetime(time_idx) + datetime.timedelta(hours=8)
    return df.rolling(datetime.timedelta(microseconds=window / 1000), min_periods=min_periods)


@jit(nogil=True)
def rolling_sum_ema_(timestamps, a, half_life):
    lambda_ = np.log(2) / half_life
    diff_time = timestamps[1:] - timestamps[:-1]
    result = np.zeros_like(a, dtype=np.float64)
    result[0] = a[0]

    for i in range(1, result.shape[0]):
        result[i] = result[i - 1] * np.exp(-lambda_ * diff_time[i - 1]) + a[i]
    return result


def rolling_sum_ema(timestamps, a, half_life):
    """
    :param timestamps:
    :param a:
    :param window: nano secs length
    :param half_life: half decay in nanos
    :return:
    """
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.values

    if hasattr(a, 'values'):
        a = a.values
    return rolling_sum_ema_(timestamps, a, half_life)


# to calculate the series begining with nan.
@jit(nogil=True)
def rolling_mean_ema_(timestamps, a, half_life):
    lambda_ = np.log(2) / half_life
    diff_time = timestamps[1:] - timestamps[:-1]
    result = np.zeros_like(a, dtype=np.float64)
    result[0] = a[0]
    for i in range(1, result.shape[0]):
        if np.isnan(a[i]):
            result[i] = np.nan
        elif np.isnan(a[i - 1]):
            result[i] = a[i]
        else:
            discount_factor = np.exp(-lambda_ * diff_time[i - 1])
            result[i] = result[i - 1] * discount_factor + (1 - discount_factor) * a[i]
    return result


def rolling_mean_ema(timestamps, a, half_life):
    if isinstance(half_life, str):
        half_life = pd.to_timedelta(half_life).value
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.values
    if hasattr(a, 'values'):
        a = a.values
    return rolling_mean_ema_(timestamps, a, half_life)


@jit(nogil=True)
def time_resample_(target_ts, ts, a):
    target_shape = list(a.shape)
    target_shape[0] = target_ts.shape[0]
    result = np.empty(target_shape, dtype=np.float64)
    ts_len = ts.shape[0]

    j = -1
    for i in range(target_shape[0]):
        while j < ts_len - 1 and ts[j + 1] <= target_ts[i]:
            j += 1
        if j == -1:
            result[i] = np.nan
        elif target_ts[i] - ts[j] > 60 * 60 * 1e9:
            result[i] = np.nan
        else:
            result[i] = a[j]
    return result


def time_resample(target_ts, ts, a):
    if hasattr(target_ts, 'values'):
        target_ts = target_ts.values

    if hasattr(ts, 'values'):
        ts = ts.values

    if hasattr(a, 'values'):
        a = a.values

    return time_resample_(target_ts, ts, a)


@jit(nogil=True)
def time_resample2_(target_ts, ts, a):
    target_shape = list(a.shape)
    target_shape[0] = target_ts.shape[0]
    result = np.empty(target_shape, dtype=np.float64)
    ts_len = ts.shape[0]

    j = -1
    for i in range(target_shape[0]):
        while j < ts_len - 1 and ts[j + 1] < target_ts[i]:
            j += 1
        if j == -1:
            result[i] = np.nan
        else:
            result[i] = a[j + 1]
    return result


def time_resample2(target_ts, ts, a):
    if hasattr(target_ts, 'values'):
        target_ts = target_ts.values

    if hasattr(ts, 'values'):
        ts = ts.values

    if hasattr(a, 'values'):
        a = a.values

    return time_resample2_(target_ts, ts, a)


@jit(nogil=True)
def quantile_mean_speeded(signal, y_pd, quantiles):
    overall_result = np.empty((y_pd.shape[1], quantiles.shape[0] - 1), dtype=np.float32)
    for i in range(y_pd.shape[1]):
        y_in = y_pd[:, i]
        for j in range(quantiles.shape[0] - 1):
            overall_result[i, j] = np.nanmean(y_in[(signal >= quantiles[j]) & (quantiles[j + 1] >= signal)])
    return overall_result


def quantile_mean(x, y, pencentile=None):
    if pencentile is None:
        pencentile = np.linspace(0, 100, 11)
    quantiles = np.percentile(x, pencentile)
    result = []
    for i in range(quantiles.shape[0] - 1):
        result.append(np.nanmean(y[(x >= quantiles[i]) & (quantiles[i + 1] >= x)]))
    return result


def quantile_std(x, y, pencentile=None):
    if pencentile is None:
        pencentile = np.linspace(0, 100, 11)
    quantiles = np.percentile(x, pencentile)
    result = []
    for i in range(quantiles.shape[0] - 1):
        result.append(np.nanstd(y[(x >= quantiles[i]) & (quantiles[i + 1] >= x)]))
    return result


def quantile_std(x, y, pencentile=None):
    if pencentile is None:
        pencentile = np.linspace(0, 100, 11)
    quantiles = np.percentile(x, pencentile)
    result = []
    for i in range(quantiles.shape[0] - 1):
        result.append(np.nanstd(y[(x >= quantiles[i]) & (quantiles[i + 1] >= x)]))
    return result


def quantile_var(x, y, pencentile=None, alpha=0.05):
    if pencentile is None:
        pencentile = np.linspace(0, 100, 11)
    quantiles = np.percentile(x, pencentile)
    result = []
    for i in range(quantiles.shape[0] - 1):
        a = y[(x >= quantiles[i]) & (quantiles[i + 1] >= x)]
        result.append(np.quantile(a[~np.isnan(a)],0.05))
    return result


def bin_mean(x, y, bins):
    result = []
    for i in range(len(bins) - 1):
        result.append(np.nanmean(y[(x >= bins[i]) & (bins[i + 1] >= x)]))
    return result


@jit(nogil=True)
def rolling_max_(ts, price, window):
    result = np.zeros_like(price)
    result_len = result.shape[0]
    top = []
    ts_top = []
    last_pop = 0
    for i in range(result_len):
        if np.isnan(price[i]):
            result[i] = np.nan
        else:
            while len(top) > 0 and top[-1] <= price[i]:
                top.pop()
                ts_top.pop()
            top.append(price[i])
            ts_top.append(ts[i])
            while ts_top[0] < ts[i] - window:
                last_pop = ts_top[0]
                top.pop(0)
                ts_top.pop(0)
            if ts[i] - last_pop > 60 * 60 * 10 ** 9:
                result[i] = np.nan
            else:
                result[i] = top[0]

    return result


def rolling_max(ts, price, window):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(price, 'values'):
        price = price.values
    return rolling_max_(ts, price, window)


@jit(nogil=True)
def rolling_min_(ts, price, window):
    result = np.zeros_like(price)
    result_len = result.shape[0]
    bot = []
    ts_bot = []
    last_pop = 0
    for i in range(result_len):
        if np.isnan(price[i]):
            result[i] = np.nan
        else:
            while len(bot) > 0 and bot[-1] >= price[i]:
                bot.pop()
                ts_bot.pop()
            bot.append(price[i])
            ts_bot.append(ts[i])
            while ts_bot[0] < ts[i] - window:
                last_pop = ts_bot[0]
                bot.pop(0)
                ts_bot.pop(0)
            if ts[i] - last_pop > 60 * 60 * 10 ** 9:
                result[i] = np.nan
            else:
                result[i] = bot[0]
    return result


def rolling_min(ts, price, window):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(price, 'values'):
        price = price.values
    return rolling_min_(ts, price, window)


@jit(nogil=True)
def rolling_mean_(ts, price, window):
    result = np.zeros_like(price)
    result_len = result.shape[0]
    j = 0
    sum = 0
    for i in range(result_len):
        if np.isnan(price[i]):
            result[i] = np.nan
        else:
            sum = sum + price[i]
        while ts[j] < ts[i] - window:
            if not np.isnan(price[j]):
                sum = sum - price[j]
            j += 1
        if np.isnan(price[j]):
            result[i] = np.nan
        else:
            result[i] = sum / (i - j + 1)
    return result


def rolling_mean(ts, price, window):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(price, 'values'):
        price = price.values
    return rolling_mean_(ts, price, window)


@jit(nogil=True)
def rolling_sum_(ts, price, window):
    result = np.zeros_like(price)
    result_len = result.shape[0]
    j = 0
    sum = 0
    for i in range(result_len):
        if np.isnan(price[i]):
            result[i] = np.nan
        else:
            sum = sum + price[i]
        while ts[j] < ts[i] - window:
            if not np.isnan(price[j]):
                sum = sum - price[j]
            j += 1
        if np.isnan(price[j]):
            result[i] = np.nan
        else:
            result[i] = sum
    return result


def rolling_sum(ts, price, window):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(price, 'values'):
        price = price.values
    return rolling_sum_(ts, price, window)


# ------------------------------------------for research-------------------------------------------


# ---------for plot------------------------------
def plot_taking_point(price, signal, percent, color=['r', 'g'], plot_price=True):
    mask1 = np.zeros_like(signal)
    mask1[np.where(signal > np.percentile(signal, percent[1]))] = 1
    mask1[np.where(signal < np.percentile(signal, percent[0]))] = -1
    df = pd.DataFrame(price)
    df['mask1'] = mask1
    if plot_price:
        plt.plot(df.mid_price, color='grey', alpha=0.4)
    plt.scatter(df[df.mask1 > 0].index, df[df.mask1 > 0].mid_price, color=color[0])
    plt.scatter(df[df.mask1 < 0].index, df[df.mask1 < 0].mid_price, color=color[1])


def plot_signal_decay2(ts, signal, y, window, bins, label_percent=False):
    y_pd = pd.DataFrame()
    for w in window:
        y_pd[w] = -time_diff(ts, y, -w * 10 ** 9)
    y_pd.index = ts
    p1 = [bin_mean(signal, y_pd[w].values, bins) for w in window]
    if label_percent:
        p1 = (np.array(p1) / y.mean()) * 100
    else:
        p1 = np.array(p1)
    p1 = pd.DataFrame(p1, index=window)
    p1.plot(figsize=(16, 8))
    plt.show()
    return


def calculate_signal_rs(ts, signals, y, window=np.arange(1, 100)):
    ps = []
    for w in window:
        z = -time_diff(ts, y, -w * 10 ** 9)
        z[np.isnan(z)] = 0
        model = linear_model.LinearRegression()
        model.fit(signals, z)
        rs = model.score(signals, z)
        ps.append(rs)
    return np.array(ps)


def plot_signal_rs(ts, signals, y, window=np.arange(1, 100)):
    signals = pd.DataFrame(signals)
    ps = calculate_signal_rs(ts, signals, y, window)
    plt.figure(figsize=(16, 8))
    plt.plot(ps, title="Signal RS Plot")
    plt.show()


def plot_signal_pnl(ts, signal, y, window, path=None):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(y, 'values'):
        y = y.values
    returns = -time_diff(ts, y, -window)
    result = []
    result0 = 0
    result.append(result0)
    day = ts[0] // window
    len0 = len(ts)
    j = 0
    while j < len0 - 1:
        while j < len0 - 1 and ts[j] < window * (day + 1):
            j += 1
        day = ts[j] // window
        if not np.isnan(returns[j] * signal[j]):
            result0 = result0 + returns[j] * signal[j]
            result.append(result0)
        j += 1
    plt.figure(figsize=(16, 8))
    plt.plot(result, title="Signal PNL Plot")
    if path is not None:
        plt.savefig(path)
    return


def plot_signal_decay_sharpe(ts, signal, y, window, quantile=np.linspace(0, 100, 11), label_percent=False, lag=None, path=None,
                      figure_num=None, ax=None):
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(y, 'values'):
        y = y.values
    if lag == 0:
        lag = None
    y_pd = pd.DataFrame()
    for w in window:
        td = -time_diff(ts, y, -w * 10 ** 9)
        if lag is not None:
            td = time_shift(ts, td, -lag)
        y_pd[w] = td
    y_pd.index = ts
    y_pd = y_pd[~np.isnan(signal)]
    y = y[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]
    if label_percent:
        p1 = [quantile_mean(signal, y_pd[w].values / y, quantile) for w in window]
        p1 = (np.array(p1))
        p1_std = [quantile_std(signal, y_pd[w].values / y.values, quantile) for w in window]
        p1_std = np.array(p1_std)
        p1_sharpe = p1/p1_std
    else:
        p1 = [quantile_mean(signal, y_pd[w].values, quantile) for w in window]
        p1 = np.array(p1)
        p1_std = [quantile_std(signal, y_pd[w].values, quantile) for w in window]
        p1_std = np.array(p1_std)
        p1_sharpe = p1/p1_std
    p1_sharpe = pd.DataFrame(p1_sharpe, index=window)
    if figure_num:
        p = plt.figure(figure_num, figsize=(8, 4))
        plt.title("Signal Decay Plot")
        plt.plot(p1_sharpe)
    else:
        if ax is None:
            p1_sharpe.plot(figsize=(8, 4), title="Signal Sharpe Decay Plot")
        else:
            p1_sharpe.plot(figsize=(8, 4), title="Signal Sharpe Decay Plot", ax=ax)
    if path is not None:
        plt.savefig(path)
    return



def plot_signal_decay_VaR(ts, signal, y, window, quantile=np.linspace(0, 100, 11), label_percent=False, lag=None, path=None,
                      figure_num=None, ax=None):
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(y, 'values'):
        y = y.values
    if lag == 0:
        lag = None
    y_pd = pd.DataFrame()
    for w in window:
        td = -time_diff(ts, y, -w * 10 ** 9)
        if lag is not None:
            td = time_shift(ts, td, -lag)
        y_pd[w] = td
    y_pd.index = ts
    y_pd = y_pd[~np.isnan(signal)]
    y = y[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]
    if label_percent:
        p1_var = [quantile_var(signal, y_pd[w].values/ y, quantile) for w in window]
        p1_var = np.array(p1_var)* 100
    else:
        p1_var = [quantile_var(signal, y_pd[w].values, quantile) for w in window]
        p1_var = np.array(p1_var)
    p1_var = pd.DataFrame(p1_var, index=window)

    if figure_num:
        p = plt.figure(figure_num, figsize=(8, 4))
        plt.title("Signal Decay Plot")
        plt.plot(p1_var)
    else:
        if ax is None:
            p1_var.plot(figsize=(8, 4), title="Signal VaR Decay Plot")
        else:
            p1_var.plot(figsize=(8, 4), title="Signal VaR Decay Plot", ax=ax)
    if path is not None:
        plt.savefig(path)
    return


def plot_signal_decay(ts, signal, y, window, quantile=np.linspace(0, 100, 11), label_percent=False, lag=None, path=None,
                      figure_num=None, ax=None):
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(y, 'values'):
        y = y.values
    if lag == 0:
        lag = None
    y_pd = pd.DataFrame()
    for w in window:
        td = -time_diff(ts, y, -w * 10 ** 9)
        if lag is not None:
            td = time_shift(ts, td, -lag)
        y_pd[w] = td
    y_pd.index = ts
    y_pd = y_pd[~np.isnan(signal)]
    y = y[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]
    if label_percent:
        p1 = [quantile_mean(signal, y_pd[w].values / y, quantile) for w in window]
        p1 = (np.array(p1)) * 100
    else:
        p1 = [quantile_mean(signal, y_pd[w].values, quantile) for w in window]
        p1 = np.array(p1)
        p1_std = [quantile_std(signal, y_pd[w].values, quantile) for w in window]
        p1_std = np.array(p1_std)
        p1_sharpe = p1/p1_std
    p1 = pd.DataFrame(p1, index=window)
    if figure_num:
        p = plt.figure(figure_num, figsize=(8, 4))
        plt.title("Signal Decay Plot")
        plt.plot(p1)
    else:
        if ax is None:
            p1.plot(figsize=(8, 4), title="Signal Decay Plot")
        else:
            p1.plot(figsize=(8, 4), title="Signal Decay Plot", ax=ax)
    if path is not None:
        plt.savefig(path)
    return


def plot_signal_decay_cache(ts, signal, y, window, quantile=np.linspace(0, 100, 11), label_percent=False, lag=None,
                            path=None, y_pd=None):
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(y, 'values'):
        y = y.values
    if lag == 0:
        lag = None
    if y_pd is None:
        y_pd = pd.DataFrame()
        for w in window:
            td = -time_diff(ts, y, -w * 10 ** 9)
            if lag is not None:
                td = time_shift(ts, td, -lag)
            y_pd[w] = td
        y_pd.index = ts
        y_pd = y_pd[~np.isnan(signal)]
    y = y[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]
    if label_percent:
        p1 = [quantile_mean(signal, y_pd[w].values / y, quantile) for w in window]
        p1 = (np.array(p1)) * 100
    else:
        p1 = [quantile_mean(signal, y_pd[w].values, quantile) for w in window]
        p1 = np.array(p1)
    p1 = pd.DataFrame(p1, index=window)
    plt.figure()
    p1.plot(figsize=(16, 8))
    if path is not None:
        plt.savefig(path)
    return y_pd


def calc_fs(prod_no, df_all, sig_name):
    from .future_utils import get_product_information, get_rolling_product
    information = get_product_information(get_rolling_product(prod_no))
    multiplier = information['point_value']

    if ''.join([i for i in prod_no if not i.isdigit()]) in ['i', 'j', 'jm', 'IC', 'IF', 'IH']:  # yestarday
        fee_ratio = information['close_ratio_by_money'] * (1 + information['broker_fee']) * (1 - information['rebate'])
        spread_offset_ratio = fee_ratio
        fee_value = information['close_ratio_by_volume'] * (1 + information['broker_fee']) * (1 - information['rebate'])
        spread_offset_value = fee_value
    else:
        fee_ratio = (information['close_ratio_by_money'] + information['close_today_ratio_by_money']) / 2 * (
                    1 + information['broker_fee']) * (1 - information['rebate'])
        spread_offset_ratio = fee_ratio
        fee_value = (information['close_ratio_by_volume'] + information['close_today_ratio_by_volume']) / 2 * (
                    1 + information['broker_fee']) * (1 - information['rebate'])
        spread_offset_value = fee_value

    df_all['forecaster'] = df_all.mid_price * (df_all[sig_name] + 1)
    fs = [1 if forecaster >= ask_price + mid_price * spread_offset_ratio + spread_offset_value / multiplier
                    else -1 if forecaster <= bid_price - mid_price * spread_offset_ratio - spread_offset_value / multiplier
                    else 0 for forecaster, ask_price, bid_price, mid_price in
                    zip(df_all.forecaster, df_all.ask_prices1, df_all.bid_prices1, df_all.mid_price)]
    fs = np.array(fs)
    return fs


def get_subpoint_different_ratio(fs_simple,fs):
    cm = confusion_matrix(fs_simple, fs)
    different_ratio = (cm[1, 0]+cm[2, 0]+cm[1, 2]+cm[0, 2])/(cm[:, 0].sum()+cm[:, 2].sum())
    return different_ratio


def plot_signal_subpoint_decay(ts, signal, y, fs, window=np.linspace(0, 60, 121), quantile=np.linspace(0, 100, 11),
                               label_percent=False, lag=None, path=None, figure_num=None):
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(y, 'values'):
        y = y.values
    if lag == 0:
        lag = None
    y_pd = pd.DataFrame()
    for w in window:
        td = -time_diff_subpoints(ts, y, -w * 10 ** 9, fs)
        y_pd[w] = td
    y_pd.index = ts
    y_pd = y_pd[~np.isnan(signal)]

    # buy
    buy_change = y_pd.iloc[np.where(fs == 1)[0]].dropna().apply(lambda x: np.percentile(x, quantile), axis=0).T
    buy_change.columns = [str(f) for f in quantile]
    # sell
    sell_change = y_pd.iloc[np.where(fs == -1)[0]].dropna().apply(lambda x: np.percentile(x, quantile), axis=0).T
    sell_change.columns = [str(f) for f in quantile]
    # total
    fs_change = pd.merge(buy_change, sell_change, suffixes=['_buy', '_sell'], left_index=True, right_index=True)

    fs_change.plot()
    plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    plt.title('each percent price change after subpoint')
    if path is not None:
        plt.savefig(path)
    plt.show()

    return


def time_diff_subpoints(timestamps, a, diff_time, fs):
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.values
    if hasattr(a, 'values'):
        a = a.values
    if hasattr(fs, 'values'):
        fs = fs.values
    diff_time = int(diff_time)

    return _time_diff_subpoints(timestamps, a, diff_time, fs)


@jit(nogil=True, nopython=True)
def _time_diff_subpoints(timestamps, a, diff_time, fs, larger=True):
    assert (a.shape[0] == timestamps.shape[0])

    result_len = a.shape[0]

    if abs(diff_time) < 1e4:
        result = np.full(result_len, np.nan)
        if diff_time > 0:
            for i in range(diff_time, result_len):
                if fs[i] == 0:
                    result[i] = np.nan
                elif timestamps[i] - timestamps[i - diff_time] > 60 * 60 * 10 ** 9:
                    result[i] = np.nan
                else:
                    result[i] = a[i] - a[i - diff_time]
        elif diff_time < 0:
            for i in range(result_len + diff_time):
                if fs[i] == 0:
                    result[i] = np.nan
                elif timestamps[i - diff_time] - timestamps[i] > 60 * 60 * 10 ** 9:
                    result[i] = np.nan
                else:
                    result[i] = a[i] - a[i - diff_time]
        else:
            result = np.zeros_like(a, dtype=np.float64)

    else:
        result = np.zeros_like(a, dtype=np.float64)
        if larger:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    if fs[i] == 0:
                        result[i] = np.nan
                        continue
                    while timestamps[i] - timestamps[j + 1] >= diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10 ** 9:
                        result[i] = a[i] - a[j]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    if fs[i] == 0:
                        result[i] = np.nan
                        continue
                    while j < result_len and timestamps[j] - timestamps[i] < -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10 ** 9:
                            result[i] = np.nan
                        else:
                            result[i] = a[i] - a[j]
                    else:
                        result[i] = np.nan

            else:
                pass
        else:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    if fs[i] == 0:
                        result[i] = np.nan
                        continue
                    while timestamps[i] - timestamps[j + 1] > diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10 ** 9:
                        result[i] = a[i] - a[j + 1]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    if fs[i] == 0:
                        result[i] = np.nan
                        continue
                    while j < result_len and timestamps[j] - timestamps[i] <= -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10 ** 9:
                            result[i] = np.nan
                        else:
                            result[i] = a[i] - a[j - 1]
                    else:
                        result[i] = np.nan

            else:
                pass

    return result


def calculate_signal_corr(ts, signal, y, window=np.arange(1, 100), lag=None, percent=None):
    """
    calculate overall correlation and correlation in each percent on each horizon
    :ts: machine_timestamp, list-like int
    :signal: signal
    :y: mid_price, list-like
    :window: hrizon list in second
    :lag: lag in second
    :percent: list-like, 0-100

    :return: array, 1st colume-overall correlation in each window,  remian columns-percent corr on each horizon
    """
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(y, 'values'):
        y = y.values

    overall_corr, overall_percent_corr = _calculate_signal_corr(ts, signal, y, window, lag, percent)

    if percent is None:
        return overall_corr.reshape(-1)
    else:
        return np.hstack([overall_corr, overall_percent_corr])


@jit(nogil=True)
def _calculate_signal_corr(ts, signal, y, window=np.arange(1, 100), lag=None, percent=None):
    
    overall_corr = np.array([0.0]*len(window))
    overall_percent_corr = []
    index_counter = 0
    for w in window:
        delta_t = int(-w * 10 ** 9)
        z = -_time_diff(ts, y, delta_t)
        if_all_nan = False
        if lag:
            z = _time_shift(ts, z, -lag)
        not_nan = ~((np.isnan(signal) | np.isnan(z)))
        not_nan_index = np.where(not_nan==True)
        z = z[not_nan_index]
        signal0 = signal[not_nan_index]    

        if signal0.std() == 0 or z.std() == 0:
            overall_corr[index_counter] = np.nan
            if_all_nan = True
        else:
            overall_corr[index_counter] = _cor_calc(signal0, z)
        index_counter += 1
        percent_corr = []
        if percent is not None:
            for i in range(len(percent) - 1):
                index = np.where((signal0 < np.percentile(signal0, percent[i + 1])) & (
                        signal0 > np.percentile(signal0, percent[i])))[0]
                if if_all_nan or len(index) == 0 or signal0[index].std() == 0 or z[index].std() == 0:
                    percent_corr.append(np.nan)
                else:
                    percent_corr.append(np.corrcoef(signal0[index], z[index])[0, 1])
        overall_percent_corr.append(percent_corr)

    overall_corr = np.array(overall_corr).reshape([-1, 1])
    overall_percent_corr = np.array(overall_percent_corr)

    return overall_corr, overall_percent_corr
    

def plot_signal_corr(ts, signal, y, window=np.arange(1, 100), lag=None, path=None, figure_num=None, ax=None):
    if lag == 0:
        lag = None
    ps = calculate_signal_corr(ts, signal, y, window, lag=lag)
    ps = pd.Series(ps, index=window)
    if figure_num:
        p = plt.figure(figure_num, figsize=(8, 4))
        plt.title("Signal Correlation Plot")
        plt.plot(ps)
    else:
        if ax is None:
            ps.plot(figsize=(8, 4), title="Signal Correlation Plot")
        else:
            ps.plot(figsize=(8, 4), title="Signal Correlation Plot", ax=ax)
    if path is not None:
        plt.savefig(path)
    return


def calculate_icir(ts, signal, y, sample_time=1, window=np.arange(1, 100), lag=None):
    if hasattr(signal, "values"):
        signal = signal.values
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(y, 'values'):
        y = y.values
    if isinstance(sample_time, datetime.timedelta):
        sample_time = int(sample_time.total_seconds() * 1e9)
    else:
        sample_time = int(86400 * 1e9 * sample_time)

    ir_w = []
    for w in window:
        delta_t = int(-w * 10 * 1e9)
        price_change = -_time_diff(ts, y, delta_t)
        if lag:
            price_change = _time_shift(ts, price_change, -lag)
        not_nan = ~(np.isnan(signal) | np.isnan(price_change))
        price_change = price_change[not_nan]
        signal0 = signal[not_nan]
        ts0 = ts[not_nan]
        time_splits = []
        last_time = ts0[0]
        for idx, t in enumerate(ts0):
            if t - last_time > sample_time:
                time_splits.append(idx)
                last_time = t
        time_splits.append(len(ts0))
        threshold_size = len(ts0) / len(time_splits) * 0.1
        #  for i in time_splits:
        #  print(pd.to_datetime(ts0[i-1]))
        ic_t = []
        prev = 0
        for i in time_splits:
            if i - prev < threshold_size:
                prev = i
                continue
            ic_t.append(np.corrcoef(signal0[prev:i], price_change[prev:i])[0, 1])
        #  print(ic_t)
        if len(ic_t) < 5:
            ir_w.append(np.nan)
        else:
            ir_w.append(np.mean(ic_t) / np.std(ic_t))
    return np.array(ir_w)


def plot_signal_corr_sqrt(ts, signal, y, window=np.arange(1, 100), lag=None, path=None):
    if lag == 0:
        lag = None
    ps = calculate_signal_corr(ts, signal, y, window, lag=lag)
    ps = pd.Series(ps * np.sqrt(window), index=window)
    ps.plot(figsize=(16, 8), title="Signal Correlation Plot")
    if path is not None:
        plt.savefig(path)

    return


def plot_signal_posi(ts, signal, y, window, quantile=np.linspace(0, 100, 11), lag=None, path=None):
    if hasattr(signal, 'values'):
        signal = signal.values
    if lag == 0:
        lag = None
    y_pd = pd.DataFrame()
    for w in window:
        td = -time_diff(ts, y, -w * 10 ** 9)
        if lag is not None:
            td = time_shift(ts, td, -lag)
        y_pd[w] = td
    y_pd.index = ts
    y_pd = y_pd[~np.isnan(signal)]
    y_pd = (y_pd > 0) * 0.5 + (y_pd >= 0) * 0.5
    signal = signal[~np.isnan(signal)]
    p1 = [quantile_mean(signal, y_pd[w].values, quantile) for w in window]
    p1 = np.array(p1)
    p1 = pd.DataFrame(p1, index=window)
    plt.figure()
    p1.plot(figsize=(16, 8), title="Signal Position Plot")
    plt.plot([0, 1000], [0.5, 0.5], 'k--')
    if path is not None:
        plt.savefig(path)
    return


def plot_signal_oneday(ts, signal, y, day, quantile=[0.01, 0.99]):
    df = pd.DataFrame()
    df['signal'] = signal
    df['y'] = y
    df.index = todatetime(ts)
    q = np.quantile(df['signal'].dropna(), quantile)
    df = df[20190000 + df.index.month * 100 + df.index.day == day]
    plt.figure(figsize=(16, 8))
    df['y'].plot(color='y', title="Signal OneDay Plot")
    plt.plot(df['y'][(df['signal'] > q[1])], 'ro', ms=5)
    plt.plot(df['y'][(df['signal'] < q[0])], 'go', ms=5)
    plt.show()
    return


def plot_signal_bar(ts, signal, quantile=[0.01, 0.99]):
    df = pd.DataFrame()
    df['signal'] = signal
    df.index = todatetime(ts)
    q = np.quantile(df['signal'].dropna(), quantile)
    plt.figure(figsize=(16, 8))
    summary = ((df['signal'] > q[1]) | (df['signal'] < q[0])).groupby(df.index.month * 100 + df.index.day).sum()
    plt.bar(summary.index.astype(str), summary)
    plt.xticks(summary.index.astype(str)[np.arange(0, len(summary), 5)])
    plt.show()
    return


def calculate_signal_beta(ts, signal, y, window=np.arange(1, 100), lag=None):
    if hasattr(signal, 'values'):
        signal = signal.values
    if lag == 0:
        lag = None

    ps = np.zeros_like(window[window != 0])
    return _calculate_signal_beta(ts, signal, y, window, lag, ps)


@jit(nopython=True)
def _calculate_signal_beta(ts, signal, y, window, lag, ps):
    for i in range(len(window)):
        w = window[i]
        if w == 0:
            continue
        signal0 = signal
        r = -log_return(ts, y, -w * 10 ** 9)
        r = r[~np.isnan(signal0)]
        signal0 = signal0[~np.isnan(signal0)]
        signal0 = signal0[~np.isnan(r)]
        r = r[~np.isnan(r)]
        ps[i] = corr(signal0, r) * np.nanstd(r)
    return ps


def plot_beta_decay(ts, signal, y, window=np.arange(1, 100), lag=None, path=None, figure_num=None):
    if lag == 0:
        lag = None
    window = window[window != 0]
    ps = calculate_signal_beta(ts.values, signal.values, y.values, window, lag=lag)
    ps = pd.Series(ps, index=window)
    if figure_num:
        p = plt.figure(figure_num, figsize=(16, 8))
        plt.title("Signal Correlation Plot")
        plt.plot(ps)
    else:
        ps.plot(figsize=(16, 8), title="Signal Correlation Plot")
    if path is not None:
        plt.savefig(path)
    return


# -------------------------------signal analyse-----------------------------------------------------------

def predict_power(ts, signal, price, log_r, shift, horizon, percentile=[1, 99], direct=False):
    bar = np.percentile(signal, percentile)
    mask = np.ones_like(signal)
    mask[(signal <= bar[0]) | (signal >= bar[1])] = 0
    if direct:
        sig_dir = np.zeros_like(signal)
        logr_dir = np.zeros_like(log_r)
        sig_dir[signal > 0] = 1
        sig_dir[signal < 0] = -1
        logr_dir[log_r > 0] = 1
        logr_dir[log_r < 0] = -1
        mask_corr = corr(sig_dir, logr_dir, mask)
    else:
        mask_corr = corr(signal, log_r, mask)

    td = -time_diff(ts, price, int(-horizon * 10 ** 9))
    if shift != 0:
        td = time_shift(ts, td, -shift)
    delta_price_std = np.nanstd(td[(signal > bar[1]) | (signal < bar[0])])

    return mask_corr * delta_price_std


def taking_point_corr(signal1, signal2, percent):
    mask1 = np.zeros_like(signal1)
    mask2 = np.zeros_like(signal2)
    mask1[np.where(signal1 > np.percentile(signal1, percent[1]))] = 1
    mask1[np.where(signal1 < np.percentile(signal1, percent[0]))] = -1
    mask2[np.where(signal2 > np.percentile(signal2, percent[1]))] = 1
    mask2[np.where(signal2 < np.percentile(signal2, percent[0]))] = -1
    return corr(mask1, mask2)


def get_corr_halflife(ts, signal, y, window=np.arange(1, 100), lag=None, percent=None, max_shiftsec=10):
    corr_array = calculate_signal_corr(ts, signal, y, window, lag, percent)
    if percent is not None:
        corr_df = pd.DataFrame(corr_array, index=window, columns=['overall'] + list(range(len(percent) - 1)))
    else:
        corr_df = pd.DataFrame(corr_array, index=window, columns=['overall'])

    max_corr = corr_df.abs().max()
    max_corr_time = corr_df.abs().idxmax()

    col_result = []
    for col in corr_df.columns:
        corr_series = corr_df[col]
        if (max_corr != corr_df.max())[col]:
            corr_series *= -1
        reversal_time = 0
        if not (corr_series < 0).all() and not (corr_series > 0).all():
            reversal_time = corr_series.idxmin()
        if reversal_time < max_corr_time[col]:
            start_time = reversal_time
        else:
            start_time = 0
        corr_decay_series = corr_series.loc[max_corr_time[col]:]
        if corr_decay_series[corr_decay_series < max_corr[col] / 2].empty:
            corr_halflife = window[-1]
        else:
            corr_halflife = corr_decay_series[corr_decay_series < max_corr[col] / 2].idxmax()
        if not pd.isnull(max_corr_time[col]):
            col_result.append([start_time, max_corr_time[col], corr_halflife, corr_df.loc[max_corr_time[col], col]])
        else:
            col_result.append([start_time, max_corr_time[col], corr_halflife, np.nan])

    result_df = pd.DataFrame(col_result, columns=['shift_sec', 'horizon', 'halflife', 'max_corr'],
                             index=corr_df.columns)
    return result_df


def calc_max_drawdown(s):
    if hasattr(s, 'values'):
        s = s.values
    index_j = np.argmax(np.maximum.accumulate(s) - s)
    index_i = np.argmax(s[:index_j])
    max_drawdown = (s[index_j] - s[index_i])
    return max_drawdown


def _calc_corr(i, trading_days, rolling_window, log_r, signal):
    logr_i = log_r.loc[str(trading_days[i]):str(trading_days[i + rolling_window-1])]
    signals_i = signal.loc[str(trading_days[i]):str(trading_days[i + rolling_window-1])]
    corr_num = signals_i.corrwith(logr_i)
    return str(trading_days[i + rolling_window-1]), corr_num


def calc_rolling_corr(time, log_r, signals, rolling_window):
    if len(signals.shape)==2:
        columns = list(range(signals.shape[1]))
    else:
        columns = [0]
    if hasattr(log_r, 'values'):
        log_r = log_r.values
    if hasattr(signals, 'columns'):
        columns = signals.columns
    if hasattr(signals, 'values'):
        signals = signals.values
    log_r = pd.Series(log_r, index=pd.to_datetime(time))
    signals = pd.DataFrame(signals, index=pd.to_datetime(time))
    trading_days = pd.unique(log_r.index.date)
    result_dic = {}
    for i in list(range(len(trading_days) - rolling_window +1)):
        k = _calc_corr(i, trading_days, rolling_window, log_r, signals)
        result_dic[k[0]] = k[1]
    rolling_corr = pd.DataFrame(result_dic).T
    rolling_corr.columns = columns
    return rolling_corr


def calc_rolling_corr_single(ts, log_r, signal, rolling_window, percent_dict=None):
    if hasattr(log_r, 'values'):
        log_r = log_r.values
    if hasattr(signal, 'values'):
        signal = signal.values

    rolling_corr_all = []
    if percent_dict is None:
        return
    for percent_i in percent_dict.keys():
        percent = percent_dict[percent_i]
        keep_index = np.where(
            (signal > np.percentile(signal, percent[0])) & (signal < np.percentile(signal, percent[1])))
        log_r_percent = pd.Series(log_r, index=pd.to_datetime(ts)).iloc[keep_index]
        signal_percent = pd.DataFrame(signal, index=pd.to_datetime(ts)).iloc[keep_index]
        trading_days = pd.unique(log_r_percent.index.date)
        result_dic = {}
        for i in list(range(len(trading_days) - rolling_window)):
            k = _calc_corr(i, trading_days, rolling_window, log_r_percent, signal_percent)
            result_dic[k[0]] = k[1]
        rolling_corr = pd.DataFrame(result_dic).T
        if not rolling_corr.empty:
            rolling_corr.columns = [percent_i]
        rolling_corr_all.append(rolling_corr)
    if len(rolling_corr_all) == 0:
        rolling_corr_all = pd.DataFrame(columns=percent_dict.keys())
    else:
        rolling_corr_all = pd.concat(rolling_corr_all, axis=1, sort=True)
    return rolling_corr_all


def t_test(rolling_corr, mean):
    t_stats = (rolling_corr.mean().abs() - mean) / rolling_corr.std() * np.sqrt(len(rolling_corr))
    p_values = pd.Series(1 - scipy.stats.t.cdf(t_stats, df=len(rolling_corr) - 1), index=rolling_corr.columns)
    result = pd.concat([t_stats, p_values], axis=1, sort=True)
    result.columns = ['t_stats', 'p_value']
    return result


def rolling_beta_daily(ts, signal, price, log_r, shift, horizon, percentile=[2, 98]):
    df = pd.DataFrame(np.vstack([signal.values, price.values, log_r.values])).T
    df.columns = ['signal', 'price', 'log_r']
    df.index = pd.to_datetime(ts) + pd.to_timedelta('8h')

    rolling_predict_power = []
    for date, group in df.groupby(df.index.date):
        signal = group.signal.values
        log_r = group.log_r.values
        price = group.price.values
        rolling_predict_power.append(
            predict_power(ts, signal, price, log_r, shift, horizon, percentile) / np.nanstd(signal))

    return rolling_predict_power


def calculate_predict_power(ts, y, signal, window=10 * 10 ** 9, thres=10):
    print(deprecate_function(signature="Tianmin"))
    quantiles = np.percentile(signal, [thres, 100 - thres])
    pct_change = -time_diff(ts, y, diff_time=-window)
    pct_change[np.isnan(pct_change)] = 0
    flags = (signal <= quantiles[0]) | (signal >= quantiles[1])
    ss = signal[flags]
    ff = pct_change[flags]
    return np.corrcoef(ss, ff)[0, 1] * np.std(ff)


def calculate_daily_power(ts, y, signal, window=10 * 10 ** 9, thres=10, draw_plots=False):
    dt = todatetime(ts)
    df = pd.DataFrame(index=dt)
    month_date = df.index.month * 100 + df.index.day
    df['ts'] = ts.values if hasattr(ts, 'values') else ts
    df['date'] = month_date
    df['y'] = y.values if hasattr(y, 'values') else y
    df['signal'] = signal.values if hasattr(signal, 'values') else signal
    daily_power = df.groupby('date').apply(
        lambda x: calculate_predict_power(x['ts'], x['y'], x['signal'], thres=thres, window=window))
    if draw_plots:
        plt.figure(figsize=(16, 8))
        plt.plot(daily_power.values)
        plt.xticks(np.arange(daily_power.shape[0]), daily_power.index)
        plt.xticks(rotation=90)
        plt.show()
    return daily_power


def calculate_daily_power2(ts, y, signal, window=10 * 10 ** 9, thres=10):
    dt = todatetime(ts)
    df = pd.DataFrame(index=dt)
    month_date = df.index.month * 100 + df.index.day
    df['ts'] = ts.values if hasattr(ts, 'values') else ts
    df['date'] = month_date
    df['y'] = y.values if hasattr(y, 'values') else y
    df['signal'] = signal.values if hasattr(signal, 'values') else signal
    pnl = -time_diff(ts, y, -window)
    df['pnl'] = pnl
    quantiles = np.percentile(signal, [thres, 100 - thres])
    df_down = df.query("signal<={}".format(quantiles[0]))
    df_up = df.query("signal>={}".format(quantiles[1]))
    down_pnl = df_down.groupby('date').agg({"pnl": "mean"})
    up_pnl = df_up.groupby('date').agg({"pnl": "mean"})
    up_pnl['down'] = down_pnl['pnl']
    up_pnl.columns = ['upper_pnl', 'down_pnl']

    return up_pnl


from py_doraemon.request import br_requests



def signal_evaluation(prod_no, ts, price, signal, window=np.linspace(0, 30, 61),
                      percent=[0, 1, 10, 30, 70, 90, 99, 100],
                      corr_bar=0.03, significance_bar=0.005, plot_graph=False, max_horizon=30,
                      horizon=None,shift=None):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(price, 'values'):
        price = price.values
    if hasattr(signal, 'values'):
        signal = signal.values
    if np.isnan(signal).any():
        warnings.warn('signal has NaN value!')
        print(f'Before dropna len:{signal.shape[0]}')
        df_temp = pd.DataFrame(np.vstack([ts, price, signal])).T.dropna()
        df_temp.columns = ['ts', 'price', 'signal']
        ts = df_temp.ts.values
        price = df_temp.price.values
        signal = df_temp.signal.values
        print(f'After dropna len:{df_temp.shape[0]}')
    if len(signal[signal == 0]) / len(signal) > 0.25:
        print(f'signal contains {round(len(signal[signal == 0]) / len(signal), 3)} 0, remove 0')
        nozero_index = np.where(signal != 0)
        ts = ts[nozero_index]
        price = price[nozero_index]
        signal = signal[nozero_index]

    if_remote = True
    try:
        br_requests.call('signal/discovery', pickle_data=dict(x=0))
        print('use remote server')
    except:
        if_remote = False
        print('use local server')
    if if_remote:
        signal_evaluation, plot_material = br_requests.call('signal/evaluation', pickle_data=dict(
            prod_no=prod_no,
            ts=ts,
            price=price,
            signal=signal,
            window=window,
            percent=percent,
            corr_bar=corr_bar,
            significance_bar=significance_bar,
            plot_graph=plot_graph,
            max_horizon=max_horizon))
    else:
        signal_evaluation, plot_material = signal_evaluation_local(prod_no, ts, price, signal, window, percent,
                                                                   corr_bar, significance_bar, plot_graph, max_horizon,horizon,shift)

    if plot_graph:
        (pecent_rollingcorr, overall_rollingcorr, robust_percentile_list, signal, log_r) = plot_material
        import seaborn as sns
        import matplotlib.colors as mcolors
        color_list = iter(list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values()))
        line_list = []
        sns.distplot(signal)
        for singl_percent in percent:
            quantile = np.percentile(signal, singl_percent)
            color = next(color_list)
            line = plt.axvline(quantile, color=color, linewidth=1)
            line_list.append(line)
        plt.legend(line_list, percent)
        plt.title('signal distribution')
        plt.show()
        pecent_rollingcorr.plot()
        plt.legend()
        plt.title('percentile rolling corr')
        plt.xticks(rotation=90)
        plt.show()
        plt.plot(overall_rollingcorr, label='overall')
        plt.title('overall rolling corr')
        plt.xticks(rotation=90)
        plt.show()
        # cunsum
        overall_cumsum = np.nan_to_num(signal * log_r, 0).cumsum()
        percent_cumsum_dict = {}
        for horizon_i in range(len(percent) - 1):
            if isinstance(horizon_i, int):
                mask = np.ones(len(signal))
                mask[(signal > np.percentile(signal, percent[horizon_i])) & (
                        signal <= np.percentile(signal, percent[horizon_i + 1]))] = 0
                percent_cumsum_dict[horizon_i] = np.nan_to_num((signal * log_r) * (1 - mask), 0).cumsum()
        percent_cumsum = pd.DataFrame(percent_cumsum_dict)
        percent_cumsum['overall'] = overall_cumsum
        (percent_cumsum / (percent_cumsum.max() - percent_cumsum.min())).plot(alpha=0.7, legend=True)
        plt.title('signal cumsum')
        plt.show()

        plot_signal_decay(ts, signal, price, window, percent)
        plt.title('signal decay')
        plt.show()

        plot_signal_corr(ts, signal, price, window)
        plt.title('signal correlation')
        plt.show()

    return signal_evaluation


def signal_evaluation_local(prod_no, ts, price, signal, window=np.linspace(0, 30, 61),
                            percent=[0, 1, 10, 30, 70, 90, 99, 100],
                            corr_bar=0.03, significance_bar=0.005, plot_graph=False, max_horizon=30,
                            horizon=None, shift=None):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(price, 'values'):
        price = price.values
    if hasattr(signal, 'values'):
        signal = signal.values
    if np.isnan(signal).any():
        warnings.warn('signal has NaN value!')
        print(f'Before dropna len:{signal.shape[0]}')
        df_temp = pd.DataFrame(np.vstack([ts, price, signal])).T.dropna()
        df_temp.columns = ['ts', 'price', 'signal']
        ts = df_temp.ts.values
        price = df_temp.price.values
        signal = df_temp.signal.values
        print(f'After dropna len:{df_temp.shape[0]}')

    signal_evaluation = {}
    all_halflife = get_corr_halflife(ts, signal, price, window=window, lag=None, percent=percent)
    if all_halflife.isnull().any(axis=1).any():
        nan_percentile = all_halflife.isnull().any(axis=1)[all_halflife.isnull().any(axis=1)].index.tolist()
        halflife = all_halflife.dropna()
    else:
        nan_percentile = []
        halflife = all_halflife
    detect_shift_sec = halflife.shift_sec.mean()
    detect_shift = int(detect_shift_sec / 0.5)
    detect_shift_sec = detect_shift * 0.5
    detect_horizon = (halflife.drop('overall', axis=0).horizon.mean() + halflife.loc['overall', 'horizon']) / 2
    detect_horizon = round(detect_horizon / 0.5) * 0.5
    percent_type = None

    # overall type
    overall_corr = halflife.loc['overall', 'max_corr']
    if overall_corr > corr_bar:
        overall_signal_type = 'strong'
    else:
        overall_signal_type = 'weak'

    # distribution type
    weak_horizon = halflife.drop('overall', axis=0)[halflife.drop('overall', axis=0).max_corr < overall_corr / 3]
    strong_horizon = halflife.drop('overall', axis=0)[halflife.drop('overall', axis=0).max_corr > overall_corr * 2]
    if not weak_horizon.empty and not strong_horizon.empty:
        percent_type = 'uneven_extrem'
    if not weak_horizon.empty:
        percent_type = 'uneven_weak'
    elif not weak_horizon.empty:
        percent_type = 'uneven_strong'
    else:
        percent_type = 'even'

    # robust check
    percent_dict = {}
    for i in range(len(percent) - 1):
        if i not in nan_percentile:
            percent_dict[i] = [percent[i], percent[i + 1]]

    if horizon is None:
        horizon = detect_horizon
    if shift is None:
        shift = detect_shift_sec
    log_r = -log_return(ts, price, -horizon * 10 ** 9, -shift*10**9)
    pecent_rollingcorr = calc_rolling_corr_single(ts, log_r, signal, rolling_window=3, percent_dict=percent_dict)
    overall_rollingcorr = calc_rolling_corr(ts, log_r, signal, rolling_window=3)
    t_test_result = t_test(pecent_rollingcorr, significance_bar)
    t_test_result_overall = t_test(overall_rollingcorr, significance_bar)
    robust_percentile_list = t_test_result[t_test_result.p_value < 1e-2].index.tolist()
    if t_test_result_overall.p_value[0] < 1e-2:
        robust_percentile_list += ['overall']

    if len(robust_percentile_list) == 0:
        IC_util = 0
        horizon_util = 0
        halflife_util = 0
    else:
        IC_util = halflife.loc[robust_percentile_list].max_corr.abs().mean()
        horizon_list = halflife.loc[robust_percentile_list].horizon
        horizon_list[horizon_list > max_horizon] = max_horizon
        horizon_util = horizon_list.mean()
        halflife_util = halflife.loc[robust_percentile_list].halflife.mean()
    signal_util = IC_util * np.sqrt(horizon_util * halflife_util)

    signal_evaluation['overall_signal_type'] = overall_signal_type
    signal_evaluation['percent_type'] = percent_type
    signal_evaluation['halflife'] = all_halflife
    signal_evaluation['robust_percentile_list'] = robust_percentile_list
    signal_evaluation['nan_percentile'] = nan_percentile
    signal_evaluation['signal_util'] = signal_util
    signal_evaluation['shift_sec'] = detect_shift_sec
    signal_evaluation['horizon'] = detect_horizon

    if plot_graph:
        plot_material = (pecent_rollingcorr, overall_rollingcorr, robust_percentile_list, signal, log_r)
    else:
        plot_material = None

    return signal_evaluation, plot_material



# ----------------for trainning------------------------

def rolling_fit_predict(clf, ts, X, y, train_day_cnt, test_day_cnt):
    dt = pd.DatetimeIndex((todatetime(ts)))
    dt_date = dt.date
    dates = np.unique(dt_date)
    dates = np.concatenate([dates, [datetime.date(2200, 1, 1)]], sort=True)  # add sentinel
    train_predict_value = np.full(shape=(X.shape[0]), fill_value=np.nan)
    test_predict_value = np.full(shape=(X.shape[0]), fill_value=np.nan)
    day_cnt = len(dates) - 1
    cur_train_start = 0
    cur_train_end = train_day_cnt
    cur_test_start = train_day_cnt
    cur_test_end = train_day_cnt + test_day_cnt
    #   print(day_cnt)
    while cur_test_start < day_cnt:
        #       print(cur_train_start, cur_train_end, cur_test_start, cur_test_end)
        train_index = (dt_date >= dates[cur_train_start]) & (dt_date < dates[cur_train_end])
        test_index = (dt_date >= dates[cur_test_start]) & (dt_date < dates[cur_test_end])
        train_X = X[train_index]
        train_y = y[train_index]
        test_X = X[test_index]
        clf.fit(train_X, train_y)
        test_predict_value[test_index] = clf.predict(test_X)
        train_predict_value[train_index] = clf.predict(train_X)
        cur_train_end += test_day_cnt
        cur_train_start += test_day_cnt
        cur_test_start += test_day_cnt
        cur_test_end += test_day_cnt
        cur_test_end = min(cur_test_end, day_cnt)
    return train_predict_value, test_predict_value


def fit_signal_for_index_future(stockIds, weights, start_time, end_time, signal_func, future_str, **kwargs):
    import LuffyAPI2 as lf
    if isnotebook():
        bar = tqdm_notebook
    else:
        bar = tqdm

    stock_signal_dict = dict()
    for stockId in bar(stockIds):
        stock_signal_dict[stockId] = signal_func(stockId, start_time, end_time, **kwargs)

    futures = lf.get_rolling_data_remote([("cffex_huaxi", future_str)], start_time, end_time)
    futures.index = todatetime(futures['machine_time_stamp'])
    futures = futures[(futures.index.hour * 100 + futures.index.minute > 931) & (
            futures.index.hour * 100 + futures.index.minute < 1455)]
    futures['mid_price'] = (futures['ask_prices1'] + futures['bid_prices1']) / 2

    preds = pd.DataFrame()
    for stockId in stockIds:
        preds[stockId] = time_resample(futures['machine_time_stamp'], np.array(stock_signal_dict[stockId].index),
                                       stock_signal_dict[stockId])
    preds = preds.fillna(0)
    preds.index = futures['machine_time_stamp']

    weights = np.array(weights)
    weights = weights / weights.sum()

    pred_weight_sum_net = preds.values.dot(weights)

    futures['signal'] = pred_weight_sum_net
    return futures


def get_stock_signal_from_seiya(stockId, start_time, end_time, feature_name, params):
    import seiya
    if stockId[0] == '6':
        exchange = 'shse'
    else:
        exchange = 'szse'
    feture = seiya.seiya_call.SeiyaFeature(stockId, feature_name, params)
    f = feture.getFeatures(start_time, end_time)
    f = f.reset_index().drop_duplicates("index", keep='last').set_index('index')[0]
    return f


def get_combine_signal(stockIds, weights, start_time, end_time, signal_func, **kwargs):
    import LuffyAPI2 as lf
    if isnotebook():
        bar = tqdm_notebook
    else:
        bar = tqdm

    stock_signal_dict = dict()
    for stockId in bar(stockIds):
        stock_signal_dict[stockId] = signal_func(stockId, start_time, end_time, **kwargs)

    idx_set = set()
    for stockId in stockIds:
        idx_set.update(stock_signal_dict[stockId].index)
    idx_set = list(idx_set)
    idx_set = np.array(sorted(idx_set))

    signal_raw = np.zeros_like(idx_set)
    for i in range(len(stockIds)):
        z = time_resample(idx_set, stock_signal_dict[stockIds[i]].index, stock_signal_dict[stockIds[i]].values)
        z[np.isnan(z)] = 0
        signal_raw = signal_raw + z * weights[i]
    signal_series = pd.Series(signal_raw, index=idx_set)
    signal = signal_series.reset_index()
    signal.columns = ['machine_time_stamps', 'signal']
    return signal


def get_combine_signal2(stockIds, weights, start_time, end_time, signal_func, **kwargs):
    import LuffyAPI2 as lf
    import heapq
    if isnotebook():
        bar = tqdm_notebook
    else:
        bar = tqdm
    resample_interval = kwargs.get('resample_interval')
    if resample_interval is None:
        resample_interval = 1
    else:
        kwargs.pop("resample_interval")

    stock_signal_dict = dict()
    for stockId in bar(stockIds):
        stock_signal_dict[stockId] = signal_func(stockId, start_time, end_time, **kwargs)

    idx_list = list(heapq.merge(*map(lambda x: stock_signal_dict[x].index, stockIds)))
    idxs = [idx_list[0]]
    for i in range(1, len(idx_list)):
        if idx_list[i] < idxs[-1] + resample_interval:
            continue
        else:
            idxs.append(idx_list[i])
    del idx_list
    idxs = np.array(idxs)
    signal_raw = np.zeros_like(idxs)
    for i in range(len(stockIds)):
        z = time_resample(idxs, stock_signal_dict[stockIds[i]].index, stock_signal_dict[stockIds[i]].values)
        z[np.isnan(z)] = 0
        signal_raw = signal_raw + z * weights[i]
    signal_series = pd.Series(signal_raw, index=idxs)
    signal = signal_series.reset_index()
    signal.columns = ['machine_time_stamps', 'signal']
    return signal


def label_return(y):
    y = np.array(y).copy()
    y[np.isnan(y)] = 0
    y[y > 0] = 1
    y[y < 0] = -1
    y[y == 0] = 0
    y = y.astype(int)
    return y


# ---------------Yimo-------------------------------
@jit(nogil=True)
def calc_window(window, ts, y, lag):
    y_pd_l = []
    for w in window:
        td = -time_diff(ts, y, int(-w * 10 ** 9))
        if lag is not None:
            td = _time_shift(ts, td, -lag)
        y_pd_l.append(td)
    return y_pd_l


def get_signal_decay(ts, signal, y, window, quantile=np.linspace(0, 100, 11), label_percent=False, lag=None, path=None):
    if hasattr(ts, 'values'):
        ts = ts.values
    if hasattr(signal, 'values'):
        signal = signal.values
    if hasattr(y, 'values'):
        y = y.values
    if lag == 0:
        lag = None

    y_pd_l = calc_window(window, ts, y, lag)
    y_pd = np.vstack(y_pd_l).T
    y_pd = y_pd[~np.isnan(signal)]

    y = y[~np.isnan(signal)]
    signal = signal[~np.isnan(signal)]

    quantiles = np.percentile(signal, quantile)
    if label_percent:
        p1 = quantile_mean_speeded(signal, y_pd / y.reshape(-1, 1), quantiles)
        p1 *= 100
    else:
        p1 = quantile_mean_speeded(signal, y_pd, quantiles)

    p1 = pd.DataFrame(p1, index=window)
    return p1


def get_forecaster_score(res, signal_col_name="signal"):
    if signal_col_name == "signal":
        decay_df = get_signal_decay(res['machine_timestamp'], res.signal, res['mid_price'],
                                    window=np.linspace(0, 30, 61),
                                    quantile=np.array([0, 1, 2, 5, 10, 90, 95, 98, 99, 100]), label_percent=False)
    else:
        decay_df = get_signal_decay(res['machine_timestamp'], res[signal_col_name], res['mid_price'],
                                    window=np.linspace(0, 30, 61),
                                    quantile=np.array([0, 1, 2, 5, 10, 90, 95, 98, 99, 100]), label_percent=False)

    # 1 Criteria (strength)
    highest = decay_df[8].max()  # This

    # 2 Criteria (strength)
    lowest = decay_df[0].min()  # This

    # 3 Criteria (ever cross)
    mask = np.where(decay_df[8] >= highest, 1, 0)
    mask = mask.cumsum().astype(bool)
    high_after_max = decay_df.loc[mask, 8]
    ever_downcross_zero = (high_after_max < 0).any()

    # 4 Criteria (ever cross)
    mask = np.where(decay_df[0] <= lowest, 1, 0)
    mask = mask.cumsum().astype(bool)
    low_after_min = decay_df.loc[mask, 0]
    ever_upcross_zero = (low_after_min > 0).any()

    # 5 Criteria (Rank IC)
    corr_list = []
    lookup_points = [0.5, 1, 2, 5, 10, 15, 20, 30]
    for h in lookup_points:
        ret1 = pd.Series(-log_return(res.machine_timestamp, res.mid_price, -h * 1e9), index=res.index)
        corr_val = res[signal_col_name].corr(ret1)
        corr_list.append(corr_val)
    max_IC = max(corr_list)
    corr = np.array(corr_list)
    mask = np.where(corr >= max_IC, 1, 0)
    mask = mask.cumsum().astype(bool)
    min_IC_after_peak = corr[mask].min()

    # 6 Criteria (Speed of realization)
    mask = np.where(decay_df[8].values >= highest, 1, 0)
    mask = mask.cumsum().astype(bool).astype(int)
    argmax_position = (1 - mask).sum()
    mask = np.where(decay_df[0].values <= lowest, 1, 0)
    mask = mask.cumsum().astype(bool).astype(int)
    argmin_position = (1 - mask).sum()

    score = 2 * (highest - lowest) - 100 * (ever_downcross_zero + ever_upcross_zero) + 10 * min_IC_after_peak + (
            1 / 30) * (argmax_position + argmin_position) / 2

    return score


def peel_correlation(corr_mat, level=0.8):
    assert corr_mat.values.shape[0] == corr_mat.values.shape[1]
    n = len(corr_mat)
    col = corr_mat.columns
    for i in range(1, n):
        for j in range(i):
            corr_val = corr_mat.values[i, j]
            if abs(corr_val) >= level:
                return (col[i], col[j], corr_val)


def select_predictors(df, predictor_cols, display_choice=True, display_graphics=False):
    scores = {}
    for p in predictor_cols:
        s = get_forecaster_score(df[["machine_timestamp", "mid_price", p]], signal_col_name=p)
        if s > 0:
            scores[p] = s
        else:
            print("%s got a negative score, and has been removed from the predictor set" % p)
    corr_mat = df[list(scores.keys())].corr()
    removal_log = []
    refined_namespace = list(scores.keys()).copy()
    while True:
        pair_to_peel = peel_correlation(corr_mat)
        if pair_to_peel is None:
            break
        log_str = ""
        score_i = scores[pair_to_peel[0]]
        score_j = scores[pair_to_peel[1]]
        log_str += "correlation coef is %s, hit i is %s, j is %s\n" % (
            pair_to_peel[2], pair_to_peel[0], pair_to_peel[1])
        if score_i > score_j:
            refined_namespace.remove(pair_to_peel[1])
            log_str += "i score is %s, j score is %s, machine choice is i" % (score_i, score_j)
        elif score_i <= score_j:
            refined_namespace.remove(pair_to_peel[0])
            log_str += "i score is %s, j score is %s, machine choice is j" % (score_i, score_j)
        else:
            assert False
        removal_log.append(log_str)
        if display_choice == True:
            print(log_str)
        if display_graphics == True:
            plot_signal_decay(df['machine_timestamp'], df[pair_to_peel[0]], df['mid_price'],
                              window=np.linspace(0, 30, 61),
                              quantile=np.array([0, 1, 2, 5, 10, 90, 95, 98, 99, 100]), label_percent=False)
            plt.suptitle("i plot, %s" % pair_to_peel[0])
            plt.show()
            plot_signal_decay(df['machine_timestamp'], df[pair_to_peel[1]], df['mid_price'],
                              window=np.linspace(0, 30, 61),
                              quantile=np.array([0, 1, 2, 5, 10, 90, 95, 98, 99, 100]), label_percent=False)
            plt.suptitle("j plot, %s" % pair_to_peel[1])
            plt.show()
            print()
        corr_mat = corr_mat[refined_namespace]
        corr_mat = corr_mat.loc[refined_namespace, :]
    other_cols = [x for x in df.columns if x not in predictor_cols]
    return df[other_cols + refined_namespace]


# ---------------Black-Scholes model---------------

def BSOptionValue(spot, strike, expiry, volatility, intRate=0, call=True):
    if np.isnan(spot) or np.isnan(strike) or np.isnan(expiry) or np.isnan(volatility):
        return np.nan
    d1 = (np.log(spot / strike) + (intRate + 0.5 * volatility * volatility) * expiry) / (volatility * np.sqrt(expiry))
    d2 = d1 - volatility * np.sqrt(expiry)
    if call:
        price = spot * (0.5 + 0.5 * erf(d1 / np.sqrt(2))) - strike * np.exp(-intRate * expiry) * (
                0.5 + 0.5 * erf(d2 / np.sqrt(2)))
    else:
        price = strike * np.exp(-intRate * expiry) * (0.5 + 0.5 * erf(-d2 / np.sqrt(2))) - spot * (
                0.5 + 0.5 * erf(-d1 / np.sqrt(2)))
    return price


def BSImpliedVolatility(spot, strike, expiry, price, intRate=0, call=True, lower=0.01, upper=1, tol=1e-6):
    if np.isnan(spot) or np.isnan(strike) or np.isnan(expiry) or np.isnan(price):
        return np.nan
    f = lambda x: BSOptionValue(spot, strike, expiry, x, intRate=intRate, call=call) - price
    if f(lower) * f(upper) > 0:
        return np.nan
    result = zero.zero(f, lower, upper, tol)
    return result


def BSDelta(spot, strike, expiry, volatility, intRate=0, call=True):
    d1 = (np.log(spot / strike) + (intRate + 0.5 * volatility * volatility) * expiry) / (volatility * np.sqrt(expiry))
    if call:
        delta = (0.5 + 0.5 * erf(d1 / np.sqrt(2)))
    else:
        delta = (0.5 + 0.5 * erf(d1 / np.sqrt(2))) - 1
    return delta


def BSVega(spot, strike, expiry, volatility, intRate=0):
    d1 = (np.log(spot / strike) + (intRate + 0.5 * volatility * volatility) * expiry) / (volatility * np.sqrt(expiry))
    vega = spot * np.sqrt(expiry) * np.exp(-0.5 * d1 * d1) / np.sqrt(
        2 * 3.141592653589793238462643383279502884197169399)
    return vega


def OptionValue_backup(spot, strike, expiry, volatility, intRate=0, call=True):
    result = np.zeros_like(spot)
    result_len = result.shape[0]
    for i in range(result_len):
        result[i] = BSOptionValue(spot[i], strike[i], expiry[i], volatility[i], intRate, call)
    return result


def OptionValue(spot, strike, expiry, volatility, intRate=0, call=True):
    if hasattr(spot, 'tolist'):
        spot = spot.tolist()
    try:
        n = len(spot)
    except TypeError:
        print('Spot must be a list, array or series.')
    else:
        if hasattr(spot, 'tolist'):
            spot = spot.tolist()
    if hasattr(strike, 'tolist'):
        strike = strike.tolist()
    elif isinstance(strike, int) or isinstance(strike, float):
        strike = [strike] * n
    if hasattr(expiry, 'tolist'):
        expiry = expiry.tolist()
    elif isinstance(expiry, int) or isinstance(expiry, float):
        expiry = [expiry] * n
    if hasattr(volatility, 'tolist'):
        volatility = volatility.tolist()
    elif isinstance(volatility, int) or isinstance(volatility, float):
        volatility = [volatility] * n

    return np.array(CBS.OptionValue(spot, strike, expiry, volatility, intRate, call))


def ImpliedVolatility_backup2(spot, strike, expiry, price, intRate=0, call=True, lower=0.01, upper=1, tol=1e-9):
    result = np.zeros_like(spot)
    result_len = result.shape[0]
    for i in range(result_len):
        if spot[i] == spot[i - 1] and expiry[i] == expiry[i - 1] and price[i] == price[i - 1] and strike[i] == strike[
            i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = BSImpliedVolatility(spot[i], strike[i], expiry[i], price[i], intRate, call, lower, upper, tol)
    return result


def ImpliedVolatility_backup(spot, strike, expiry, price, intRate=0, call=True, lower=0.01, upper=1, tol=1e-9):
    result = np.zeros_like(spot)
    result_len = result.shape[0]
    for i in range(result_len):
        if spot[i] == spot[i - 1] and expiry[i] == expiry[i - 1] and price[i] == price[i - 1] and strike[i] == strike[
            i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = CBS.BSImpliedVolatility(spot[i], strike[i], expiry[i], price[i], intRate, call, lower, upper,
                                                tol)
    return result


def ImpliedVolatility(spot, strike, expiry, price, intRate=0, call=True, lower=0.01, upper=1, tol=1e-9):
    if hasattr(spot, 'tolist'):
        spot = spot.tolist()
    try:
        n = len(spot)
    except TypeError:
        print('Spot must be a list, array or series.')
    if hasattr(strike, 'tolist'):
        strike = strike.tolist()
    elif isinstance(strike, int) or isinstance(strike, float):
        strike = [strike] * n
    if hasattr(expiry, 'tolist'):
        expiry = expiry.tolist()
    elif isinstance(expiry, int) or isinstance(expiry, float):
        expiry = [expiry] * n
    if hasattr(price, 'tolist'):
        price = price.tolist()
    elif isinstance(price, int) or isinstance(price, float):
        price = [price] * n

    return np.array(CBS.ImpliedVolatility(spot, strike, expiry, price, intRate, call, lower, upper, tol))


def Delta_backup2(spot, strike, expiry, volatility, intRate=0, call=True):
    result = np.zeros_like(spot)
    result_len = result.shape[0]
    for i in range(result_len):
        if spot[i] == spot[i - 1] and expiry[i] == expiry[i - 1] and volatility[i] == volatility[i - 1] and strike[i] == \
                strike[i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = BSDelta(spot[i], strike[i], expiry[i], volatility[i], intRate, call)
    return result


def Delta_backup(spot, strike, expiry, volatility, intRate=0, call=True):
    result = np.zeros_like(spot)
    result_len = result.shape[0]
    for i in range(result_len):
        if spot[i] == spot[i - 1] and expiry[i] == expiry[i - 1] and volatility[i] == volatility[i - 1] and strike[i] == \
                strike[i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = CBS.BSDelta(spot[i], strike[i], expiry[i], volatility[i], intRate, call)
    return result


def Delta(spot, strike, expiry, volatility, intRate=0, call=True):
    if hasattr(spot, 'tolist'):
        spot = spot.tolist()
    try:
        n = len(spot)
    except TypeError:
        print('Spot must be a list, array or series.')
    if hasattr(strike, 'tolist'):
        strike = strike.tolist()
    elif isinstance(strike, int) or isinstance(strike, float):
        strike = [strike] * n
    if hasattr(expiry, 'tolist'):
        expiry = expiry.tolist()
    elif isinstance(expiry, int) or isinstance(expiry, float):
        expiry = [expiry] * n
    if hasattr(volatility, 'tolist'):
        volatility = volatility.tolist()
    elif isinstance(volatility, int) or isinstance(volatility, float):
        volatility = [volatility] * n

    return np.array(CBS.Delta(spot, strike, expiry, volatility, intRate, call))


def Vega2(spot, strike, expiry, volatility, intRate=0):
    result = np.zeros_like(spot)
    result_len = result.shape[0]
    for i in range(result_len):
        if spot[i] == spot[i - 1] and expiry[i] == expiry[i - 1] and volatility[i] == volatility[i - 1] and strike[i] == \
                strike[i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = BSVega(spot[i], strike[i], expiry[i], volatility[i], intRate)
    return result


def Vega(spot, strike, expiry, volatility, intRate=0):
    result = np.zeros_like(spot)
    result_len = result.shape[0]
    for i in range(result_len):
        if spot[i] == spot[i - 1] and expiry[i] == expiry[i - 1] and volatility[i] == volatility[i - 1] and strike[i] == \
                strike[i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = CBS.BSVega(spot[i], strike[i], expiry[i], volatility[i], intRate)
    return result


def Expiry_backup(beg, end):
    from py_doraemon import trading_days

    result = np.zeros_like(beg, dtype=np.double)
    result_len = result.shape[0]
    result[0] = len(trading_days.trading_days_between(beg[0], end)) / 250
    for i in range(1, result_len):
        if beg[i] == beg[i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = len(trading_days.trading_days_between(beg[i], end)) / 250
    return result


def Expiry(beg, end):
    if hasattr(beg, 'tolist'):
        beg = beg.tolist()
    try:
        n = len(beg)
    except TypeError:
        print('Spot must be a list, array or series.')
    if hasattr(end, 'tolist'):
        end = end.tolist()
    elif isinstance(end, int) or isinstance(end, float):
        end = [end] * n
    return np.array(CBS.Expiry(beg, end))
