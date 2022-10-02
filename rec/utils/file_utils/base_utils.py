import logging
import os
import pathlib
import random
import re
import shutil
import threading
import time
import typing

import arrow

from .. import func_tools
from .. import performance
from .. import shell


def make_parent_dirs(file_path, parent=True):
    if parent:
        dirname = os.path.dirname(file_path)
    else:
        dirname = file_path
    if dirname:  # ./data.txt：FileNotFoundError: [Errno 2] No such file or directory: ''
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    return dirname


def is_valid_file(file_path, *,
                  mb_size=None, bytes_size=None,
                  num_line=None, min_lines=None,
                  expire_seconds=None):
    if mb_size is not None and bytes_size is not None:
        raise ValueError("Only one value of mb_size and bytes_size can be set to None")
    if not os.path.isfile(file_path):
        return False
    if num_line is None:
        num_line = 1
    if min_lines is None:
        min_lines = 1
    #
    flag_size = flag_line = flag_seconds = False
    # 文件行数
    line_limit = max(num_line, min_lines)
    if line_limit is None:
        flag_line = True
    else:
        try:
            for _n_line, _ in enumerate(open(file_path), start=1):
                if line_limit >= _n_line:
                    flag_line = True
                    break
        except UnicodeDecodeError:  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
            flag_line = True
    # 文件大小
    if mb_size is None and bytes_size is None:
        flag_size = True
    else:
        if bytes_size is None:
            bytes_size = mb_size * 1024 * 1024
        if os.stat(file_path).st_size >= bytes_size:
            flag_size = True
    # 文件时间
    if expire_seconds is None:
        flag_seconds = True
    else:
        if (arrow.now() - arrow.get(int(os.path.getmtime(file_path)))).seconds < expire_seconds:
            flag_seconds = True
    flag = all([flag_size, flag_line, flag_seconds])
    return flag


def is_valid_files(file_paths, /, *, mb_size=None, bytes_size=None, num_line=1, expire_seconds=None):
    if isinstance(file_paths, (str, pathlib.Path)):
        file_paths = [file_paths]
    flags = [is_valid_file(file_path, mb_size=mb_size, bytes_size=bytes_size,
                           num_line=num_line,
                           expire_seconds=expire_seconds)
             for file_path in file_paths]
    return all(flags)


@performance.timing()
def get_num_lines(file_name):
    if not os.path.isfile(file_name):
        raise FileNotFoundError(file_name)
    try:
        stdout_str = shell.shell(f"wc -l {file_name}", verbose=False).output()
        num_line = int(stdout_str.split()[0])
        return num_line
    except Exception:
        pass
    num_line = sum([1 for _ in open(file_name)])
    return num_line


# def get_path_size(file_path, unit='MB'):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(file_path)
#     stdout_str = shell.shell(f"du -s {file_path}", verbose=False).output()
#     num_kb = int(stdout_str.split()[0])
#     memo = func_tools.byte2human(bytes=num_kb * 1024, unit=unit)
#     return memo


def get_file_size(file_path, unit='MB'):
    memo = func_tools.byte2human(os.stat(file_path).st_size, unit=unit)
    return memo


# @performance.timing()
def merge_file(src_paths, dst_file_path, mode='w'):
    if isinstance(src_paths, str):
        src_paths = [src_paths]
    all_src_paths = []
    for src_path in src_paths:
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        if os.path.isfile(src_path):
            all_src_paths.append(src_path)
        else:
            all_src_paths.append(src_path.rstrip('/') + '/*')
            # for path in pathlib.Path(src_path).rglob('*'):
            #     all_src_paths.append(path.as_posix())
    make_parent_dirs(dst_file_path)
    all_src_paths = sorted(all_src_paths)
    redirect = {'w': '>', 'a': '>>'}[mode]
    cmd = f"cat {' '.join(all_src_paths)} {redirect} {dst_file_path}"
    sh = shell.shell(cmd, shell=True)
    return sh.code


def append(src_path, dst_path):
    merge_file(src_path, dst_path, mode='a')


def split_file(file_path, dst_dir=None,
               out_file_count=None, line_count: int = None, unit_size: str = None,
               verbose=False):
    """
        # -a, --suffix-length=N   use suffixes of length N (default 2)  输出文件后缀长度，默认为：2
        # -b, --bytes=SIZE        put SIZE bytes per output file  按照文件大小分割文件，单位：字节
        #               SIZE may be (or may be an integer optionally followed by) one of following:（输出文件大小也可以用以下单位）
        #               KB 1000, K 1024, MB 1000*1000, M 1024*1024, and so on for G, T, P, E, Z, Y.
        #  -l, --lines=NUMBER      put NUMBER lines per output file  按照行数分割文件，默认1000行一个文件
        # --verbose print a diagnostic just before each output file is opened 打印运行状态信息
        -------
        #  Mandatory arguments to long options are mandatory for short options too.
        #     -a, --suffix-length=N   use suffixes of length N (default 2)
        #     -b, --bytes=SIZE        put SIZE bytes per output file
        #     -C, --line-bytes=SIZE   put at most SIZE bytes of lines per output file
        #     -d, --numeric-suffixes  use numeric suffixes instead of alphabetic
        #     -l, --lines=NUMBER      put NUMBER lines per output file
        #     --verbose           print a diagnostic just before each  output file is opened
        #     --help     display this help and exit
        #     --version  output version information and exit
        #
        #   SIZE may be (or may be an integer optionally followed by) one of following:
        #   KB 1000, K 1024, MB 1000*1000, M 1024*1024, and so on for G, T, P, E, Z, Y.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
    # 切分单元
    if line_count is None and unit_size is None and out_file_count is None:
        raise ValueError(f"Exactly one of line_count({line_count}) / unit_size({unit_size}) / "
                         f"out_file_count({out_file_count}) be specified")
    #
    if unit_size is not None:
        per_file_bytes = func_tools.human2bytes(unit_size)
        out_file_count = round(os.stat(file_path).st_size / per_file_bytes) + 1
    if out_file_count is not None:
        line_count = round(get_num_lines(file_path) / out_file_count)
    cmd = f"split -a 4 -l {line_count} {file_path}"
    #
    sh = shell.shell(cmd, verbose=verbose)
    if sh.code != 0:
        raise Exception(sh.errors())
    #
    # 移动到目标文件夹
    file_name = file_path.split('/')[-1]
    if dst_dir is None:
        dst_dir = get_temp_unique_dir(f"split_{file_name}")
    os.makedirs(dst_dir, exist_ok=True)
    dst_paths = []
    for pather in pathlib.Path('./').glob('*'):
        stem = pather.stem
        if len(stem) == 5 and stem.startswith('x'):
            _file_path = pather.as_posix()
            diff = arrow.now() - arrow.get(int(os.path.getmtime(_file_path)))
            if diff.seconds < 10 * 60:
                dst_path = os.path.join(dst_dir, pather.name)
                shutil.move(_file_path, dst_path)
                dst_paths.append(dst_path)
    return dst_paths
    # mac
    # if platform.system() == 'Darwin':
    #     if unit_size is not None:
    #         byte_count = func_tools.human2bytes(unit_size)
    #         out_file_count = os.stat(file_path).st_size / byte_count
    #     if out_file_count is not None:
    #         line_count = get_num_lines(file_path) / out_file_count
    # else:  # linux
    #     if unit_size is not None:
    #         param = f'-C {byte_count}'
    #     else:
    #         param = f'-l {line_count}'
    #     cmd = f"split -a 4 {param} {file_path}"


def get_temp_unique_dir(unique_str=''):
    with threading.Lock() as lock:
        for ss in [arrow.now().format("YYYYMMDD"),
                   arrow.now().format("YYYYMMDD-hh"),
                   arrow.now().format("YYYYMMDD-hhmm"),
                   arrow.now().format("YYYYMMDD-hhmmss")]:
            tmp_dir = os.path.join('temp', unique_str, ss)
            if not os.path.exists(tmp_dir):
                return tmp_dir
            else:
                time.sleep(random.randint(0, 3))


def delete(file_path: str, max_sub_files_count=100, file_not_found_ok=True, reason='', verbose=False):
    """
    :param file_path: 待删除路径
    :param max_sub_files_count: 一个目录下最多包含的文件数，过多可能误删；建议删除其对应的子目录
    :param file_not_found_ok: 当文件路径不存在时，删除操作是否报错
    :param verbose:
    :return:
    """
    # handle_warn = '不允许自动删除，请仔细检查然后手动删除'
    if file_path.startswith('~'):
        print(f"请使用绝对路径！,不支持使用`~`表示home目录； {file_path}")
        raise ValueError(file_path)
    if re.findall('\s', file_path):
        print(f"文件路径包含不可见字符：{file_path}")
        raise ValueError(file_path)
    file_path = os.path.abspath(file_path)
    if file_path in ['/', '~', '/home', '/root'] or file_path.count('/') <= 2 or len(file_path) < 5:
        print(f"高风险路径，存在误删风险，不允许自动删除，请仔细检查然后手动删除：{file_path}")
        raise PermissionError
    if not os.path.exists(file_path):
        if file_not_found_ok:
            if verbose:
                print(f"文件不存在，无法删除：{file_path}")
            return
        raise FileNotFoundError(file_path)
    #
    if os.path.isfile(file_path):
        # os.remove(file_path)  # 要等进程结束才真正释放删除的空间
        remove(file_path, verbose=False)
    else:  # 目录
        if not ({'temp', 'tmp'} & set(file_path.split('/'))):  # 对于很大的非零时文件，需要手动删除; 零时文件不适用此规则
            all_dir_file_paths = list(pathlib.Path(file_path).rglob('*'))
            file_count = len(all_dir_file_paths)
            if file_count >= max_sub_files_count:
                print(f"此文件夹下的文件数目:{file_count}>{max_sub_files_count}，存在误删风险，不允许自动删除。"
                      f"请仔细检查然后手动删除：{file_path}")
                return
        # os.removedirs(file_path) # Directory not empty:
        # shutil.rmtree(file_path)  # 要等进程结束才真正释放删除的空间
        remove(file_path, verbose=False)
    if verbose:
        reason_str = ', reason: ' if reason else ''
        print(f"文件删除成功:{file_path} {reason_str}")
    logging.info(f"delete file_path: {file_path}")
    return True


def remove(file_path, verbose=False):
    cmd = f"rm -rf {file_path}"
    sh = shell.shell(cmd, shell=True, verbose=verbose)
    is_success = (sh.code == 0)
    return is_success


def get_var_abbrev_name(variable):
    def get_stem(v):
        name = ''
        if isinstance(v, (str, int, float)):
            name = pathlib.Path(str(v)).name
        return name

    var_str_name = ''
    if isinstance(variable, (str, int, float)):
        var_str_name = get_stem(variable)
    elif isinstance(variable, dict):
        var_str_name = '~'.join(sorted([f"{get_stem(k)}:{get_stem(v)}" for k, v in variable.items()]))
    elif isinstance(variable, typing.Iterable):
        var_str_name = '~'.join([get_stem(s) for s in variable])
    return var_str_name
