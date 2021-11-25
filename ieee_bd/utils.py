import re
from pytorch_model_summary import summary
import numpy as np


def time_to_str(t_seconds):
    """Express time in string"""
    t_seconds = int(round(t_seconds))
    if t_seconds // (365*24*60*60):
        f_str = 'year:day:hr:mins:secs'
    elif t_seconds // (24*60*60):
        f_str = 'day:hr:mins:secs'
    else:
        f_str = 'hr:mins:secs'

    value_ = {'year': t_seconds // (60*60*24*365),
              'day': (t_seconds // (60*60*24)) % 365,
              'hr': (t_seconds // (60*60)) % 24,
              'mins': (t_seconds // 60) % 60,
              'secs': t_seconds % 60,
              }
    format_ = {'year': 1,
               'day': 3,
               'hr': 2,
               'mins': 2,
               'secs': 2,
              }

    f_split = f_str.split(':')
    result = ':'.join([f"{value_[x]:0{format_[x]}d}" for x in f_split])
    return result


def compute_padding(size_out, size_in, kernel_size, stride, padding='same'):
    pad_size = 0
    if padding == 'same':
        pad_size = (stride * (size_out - 1) + kernel_size - size_in) / 2
    return int(np.ceil(pad_size))


def model_summary(model, inputs, print_summary=False, max_depth=1, show_parent_layers=False):
    # _ = summary(model, x_in, print_summary=True)
    kwargs = {'max_depth': max_depth,
              'show_parent_layers': show_parent_layers}
    sT = summary(model, inputs, show_input=True, print_summary=False, **kwargs)
    sF = summary(model, inputs, show_input=False, print_summary=False, **kwargs)

    st = sT.split('\n')
    sf = sF.split('\n')

    sf1 = re.split(r'\s{2,}', sf[1])
    out_i = sf1.index('Output Shape')

    ss = []
    i_esc = []
    for i in range(0, len(st)):
        if len(re.split(r'\s{2,}', st[i])) == 1:
            ssi = st[i]
            if len(set(st[i])) == 1:
                i_esc.append(i)
        else:
            sfi = re.split(r'\s{2,}', sf[i])
            sti = re.split(r'\s{2,}', st[i])
            # ptr = st[i].index(sti[2]) + len(sti[2])
            # in_1 = sf[i].index(sfi[1]) + len(sfi[1])
            # in_2 = sf[i].index(sfi[2]) + len(sfi[2])

            ptr = st[i].index(sti[out_i]) + len(sti[out_i])
            in_1 = sf[i].index(sfi[out_i-1]) + len(sfi[out_i-1])
            in_2 = sf[i].index(sfi[out_i]) + len(sfi[out_i])
            ssi = st[i][:ptr] + sf[i][in_1:in_2] + st[i][ptr:]
        ss.append(ssi)

    n_str = max([len(s) for s in ss])
    for i in i_esc:
        ss[i] = ss[i][-1] * n_str

    ss = '\n'.join(ss)
    if print_summary:
        print(ss)

    return ss

