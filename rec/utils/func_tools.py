import collections
import itertools
import random
import re
import typing


def flatten_dict(d, parent_key='', sep='.'):
    # from pandas.io.json._normalize import nested_to_record
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


unit2bytes = {'B': 1,
              'KB': 1024,
              'MB': 1024 * 1024, 'M': 1024 * 1024,
              'GB': 1024 * 1024 * 1024, 'G': 1024 * 1024 * 1024,
              'TB': 1024 * 1024 * 1024 * 1024, 'T': 1024 * 1024 * 1024 * 1024
              }


def byte2human(bytes, unit='B', precision=2):
    unit = unit.upper()
    memo = bytes / unit2bytes[unit]
    memo = round(memo, precision)
    return memo


def human2bytes(mem_str):
    mem_str = mem_str.upper()
    matchs = re.findall('^(\d+)([A-Z]+)$', mem_str)
    if len(matchs) != 1:
        raise ValueError(mem_str)
    size, unit = matchs[0]
    bytes_size = int(size) * unit2bytes[unit]
    return bytes_size


def invert_map_kv(dictionary: typing.Dict[typing.Hashable, typing.Hashable]):
    reversed_dictionary = {value: key for (key, value) in dictionary.items()}
    return reversed_dictionary


def get_uniq_list(li):
    new_li = sorted(set(li), key=li.index)  # 去重，且保持顺序
    # new_li = []
    # d = set()
    # for e in li:
    #     if e not in d:
    #         new_li.append(e)
    #         d.add(e)
    return new_li


def get_batch_count(iterable, batch_size):
    quotient, remainder = divmod(len(iterable), batch_size)
    batch_cnt = quotient if not remainder else quotient + 1
    return batch_cnt


def batchnize(iterable, batch_size) -> typing.Generator:
    batch = []
    for e in iterable:
        batch.append(e)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def object2dict(obj):
    data = {key: value for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('__')}
    return data


def get_random_key(d: typing.Dict):
    target_pos = random.randint(1, min(1000, len(d) - 1))
    for i, key in enumerate(d):
        if i == target_pos:
            return key


def sample_kv(d: typing.Dict, n=5):
    positions = random.sample(range(min(10000, len(d) - 1)), k=n)
    positions = set(positions)
    result = {}
    for i, key in enumerate(d):
        if i in positions:
            result[key] = d[key]
            if len(result) >= n:
                break
    return result
