import functools
import logging
import os
import threading
import time
import traceback

import psutil

from rec.utils.func_tools import byte2human


class Timeit(object):
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("time: {:.3f}".format(time.time() - self.start_time))


def timing(func=None, threshold_seconds=5, prefix=None):
    def out_wrapper(func):
        nonlocal prefix
        if prefix is None:
            module_name = func.__module__
            func_name = func.__name__
            prefix = "{}.{}".format(module_name, func_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            total_time = time.time() - start_time
            if total_time >= threshold_seconds:
                print(f'{prefix} take time: {total_time:.1f}s')
                # source_code = inspect.getsource(func).strip('\n')
                # print(source_code + ":  " + str(total_time) + " seconds")
            return res

        return wrapper

    if func is None:
        return out_wrapper  # @try_catch_with_logging()
    return out_wrapper(func)  # @try_catch_with_logging


class ShowTime(object):
    '''
    用上下文管理器计时
    '''

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __enter__(self):
        self.start_timestamp = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runtime = time.time() - self.start_timestamp
        print("{} take time: {:.2f} s".format(self.prefix, self.runtime))
        if exc_type is not None:
            print(exc_type, exc_val, exc_tb)
            print(traceback.format_exc())
            return self


class ProcessManager(object):
    def __init__(self, check_secends=20, memo_unit='GB', precision=2):
        self.pid = os.getpid()
        self.p = psutil.Process(self.pid)
        self.check_secends = check_secends
        self.memo_unit = memo_unit
        self.precision = precision
        self.start_time = time.time()

    def kill(self):
        child_poll = self.p.children(recursive=True)
        for p in child_poll:
            if not 'SYSTEM' in p.username:
                print(f'kill sub process: PID: {p.pid}  user: {p.username()} name: {p.name()}')
                p.kill()
        self.p.kill()
        print(f'kill {self.pid}')

    def get_memory_info(self):
        memo = byte2human(self.p.memory_info().rss, self.memo_unit)
        info = psutil.virtual_memory()
        total_memo = byte2human(info.total, self.memo_unit)
        used = byte2human(info.used, self.memo_unit)
        free = byte2human(info.free, self.memo_unit)
        available = byte2human(info.available, self.memo_unit)
        cur_pid_percent = info.percent
        return memo, used, free, available, total_memo, cur_pid_percent

    def task(self):
        while True:
            memo, used, free, available, total_memo, cur_pid_percent = self.get_memory_info()
            print('--' * 20)
            print(f'PID: {self.pid} name: {self.p.name()}')
            print(f'当前进程内存占用 :\t {memo:.2f} {self.memo_unit}')
            print(f'used           :\t {used:.2f} {self.memo_unit}')
            print(f'free           :\t {free:.2f} {self.memo_unit}')
            print(f'total          :\t {total_memo} {self.memo_unit}')
            print(f'内存占比        :\t {cur_pid_percent} %')
            print(f'运行时间        :\t {(time.time() - self.start_time) / 60:.2f} min')
            # print('cpu个数：', psutil.cpu_count())
            if cur_pid_percent > 95:
                logging.info(f'内存占比过高: {cur_pid_percent}%， kill {self.pid}')
                self.kill()  # 停止进程
            time.sleep(self.check_secends)

    def run(self):
        thr = threading.Thread(target=self.task)
        thr.setDaemon(True)  # 跟随主线程结束
        thr.start()
