import functools
import hashlib
import json
import logging
import os
import pathlib
import pickle
import re
import traceback
import typing

import orjson  # 速度快很多>3倍
import tqdm

from .. import performance
from .base_utils import make_parent_dirs, get_num_lines, is_valid_file, get_var_abbrev_name


@performance.timing()
def pkl_load(file_path: str, verbose=True) -> typing.Union[typing.Dict, typing.Sequence]:
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    if verbose:
        if hasattr(obj, '__len__') and callable(getattr(obj, '__len__')):
            print(f'load pkl (length={len(obj)}) from : {file_path}')
        else:
            print('load pkl from : {}'.format(file_path))
    return obj


@performance.timing()
def pkl_dump(obj: object, file_path: str, verbose=True, **kwargs):
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    make_parent_dirs(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, **kwargs)
    if verbose:
        print('save object to: {}'.format(file_path))


@performance.timing()
def json_load(file_path, verbose=True):
    with open(file_path, 'rb') as f:
        content = f.read()
        obj = orjson.loads(content)
    if verbose:
        print(f'load json (length={len(obj)}) from : {file_path}')
    return obj


@performance.timing()
def json_dump(dict_data, save_path, override_exist=True, verbose=True):
    if override_exist or not os.path.isfile(save_path):
        try:
            json_str = orjson.dumps(dict_data, option=orjson.OPT_INDENT_2)
            open_mode = 'wb'
        except TypeError as e:
            print(f"orjson error: {e}, switch to json")
            json_str = json.dumps(dict_data, ensure_ascii=False, indent=2)
            open_mode = "w"
        # strs = json.dumps(dict_data, indent=2, ensure_ascii=False)
        # strs = bytes(strs, encoding='utf-8')
        # dirname = os.path.dirname(save_path)
        # if dirname:  # ./data.txt：FileNotFoundError: [Errno 2] No such file or directory: ''
        #     os.makedirs(dirname, exist_ok=True)
        make_parent_dirs(save_path)
        with open(save_path, mode=open_mode) as f:
            f.write(json_str)
            # if save_memory:
            #     json.dump(dict_data, f, ensure_ascii=False)
            # else:
            #     json.dump(dict_data, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    if verbose:
        print('save json to: {}'.format(save_path))


def save_data(data, save_path, verbose=True):
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    make_parent_dirs(save_path)
    suffix = pathlib.Path(save_path).suffix
    if suffix in ['.pkl']:
        pkl_dump(data, save_path, verbose=verbose)
    elif suffix in ['.json']:
        json_dump(data, save_path, verbose=verbose)
    else:
        raise TypeError(suffix)


def load_data(save_path, verbose=True):
    suffix = pathlib.Path(save_path).suffix
    if suffix in ['.pkl']:
        data = pkl_load(save_path, verbose=verbose)
    elif suffix in ['.json']:
        data = json_load(save_path, verbose=verbose)
    else:
        raise TypeError(suffix)
    return data


@performance.timing()
def tqdm_iter_txt(file_path, encoding='utf-8', desc='', desc_prefix_depth=2):
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xaa in position 3968: invalid start byte
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    line_num = get_num_lines(file_path)
    pather = pathlib.Path(file_path)
    names = []
    for _ in range(desc_prefix_depth):
        names.insert(0, pather.name)
        pather = pather.parent
    stem = '/'.join(names)
    if stem not in desc:
        desc = f"read {desc} {stem}"
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line in tqdm.tqdm(f, total=line_num, desc=desc):
                yield line
    except UnicodeDecodeError:
        logging.info(f"{traceback.format_exc()}"
                     f"使用utf-8读取文件失败，尝试使用默认编码(encoding=None)读取文件: {file_path}")
        with open(file_path, 'r', encoding=None) as f:
            for line in tqdm.tqdm(f, total=line_num, desc=desc):
                yield line


@performance.timing()
def write_lines(lines, file_path):
    make_parent_dirs(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.rstrip('\n') + '\n'
            f.write(line)


# mem = joblib.Memory(location='cache', compress=True, verbose=1) # bestway
def cache(func=None, /, *, file_type=".pkl",
          cache_file_name=None, sub_cache_dir=None,
          use_parsms=False,
          min_cache_mb_size=0,
          verbose=True
          ):
    """
    :param func:
    :param / 斜杠位置以前，只能接受位置参数，不能接受关键字参数(func=...) ; python3.8
    :param * 只接受关键字参数，不接受位置参数
    :param min_cache_mb_size 超过才缓存，太小的文件不缓存
    :param use_common_cache_dir 不使用 环境变量 unique_cache_dir
    """
    file_type = file_type.strip('.')
    cache_file_name = cache_file_name if cache_file_name is not None else ''
    use_cache = (os.environ.get('use_cache', '') == 'true')  # 系统环境变量设置
    cache_dir = './cache'
    if sub_cache_dir is None:
        sub_cache_dir = os.environ.get('sub_cache_dir', '')
        cache_dir = os.path.join(cache_dir, sub_cache_dir)

    def decorate(func):
        prefix = f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not use_cache:
                if verbose:
                    print(f"{prefix} 设置不使用cache")
                return func(*args, **kwargs)

            #
            def get_params_str():
                params_str = ''
                if use_parsms:
                    # os.pathconf('/', 'PC_PATH_MAX')  # 最大路径长度
                    # os.pathconf('/', 'PC_NAME_MAX')  # 最大文件名长度
                    args_names = [get_var_abbrev_name(arg) for arg in args]
                    arg_str = ','.join([e for e in sorted(args_names) if e])
                    kv_str = get_var_abbrev_name(kwargs)
                    params_str = f"(args={arg_str})"
                    if kv_str:
                        params_str = f"(args={arg_str},kwargs={kv_str})"
                    if len(params_str) > 100:
                        params_str = f".({hashlib.md5(params_str.encode('utf-8')).hexdigest()})"
                return params_str

            def print_data_info(action='save'):
                direction = {"save": "to", "load": "from"}[action]
                if hasattr(data, '__len__') and callable(getattr(data, '__len__')):
                    print(f'{action} cache {file_type} (length={len(data)}) {direction} : {cache_file_path}')
                else:
                    print(f'{action} cache {file_type} {direction} : {cache_file_path}')

            #
            nonlocal use_parsms, cache_file_name
            params_str = get_params_str()
            cache_file_name = f'{prefix}.{cache_file_name}{params_str}.{file_type}'
            cache_file_name = re.sub(r'([.-]){2,}', r'\1', cache_file_name)  # 重复的-或者.只保留一个
            cache_file_path = os.path.join(cache_dir, cache_file_name)
            #
            if not is_valid_file(cache_file_path, mb_size=min_cache_mb_size):
                data = func(*args, **kwargs)
                save_data(data, cache_file_path, verbose=False)
                if verbose:
                    print_data_info(action='save')
            data = load_data(cache_file_path, verbose=False)
            if verbose:
                print_data_info(action="load")
            return data

        return wrapper

    if func is None:
        return decorate  # @cache()
    return decorate(func)  # @cache
