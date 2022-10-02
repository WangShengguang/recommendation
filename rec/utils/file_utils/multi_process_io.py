import concurrent.futures
import logging
import multiprocessing
import os
import queue
import time
import traceback
import typing

import psutil
import tqdm

from .base_utils import get_num_lines, split_file, delete


class Producer(multiprocessing.Process):
    def __init__(self, in_queue, file_path, read_func=None):
        super(Producer, self).__init__()
        self.in_queue: multiprocessing.Queue = in_queue
        self.file_path: multiprocessing.Event = file_path
        if read_func is None:
            read_func = open
        self.read_func = read_func

    def run(self):
        for line in self.read_func(self.file_path):
            self.in_queue.put(line)


class Consumer(multiprocessing.Process):
    def __init__(self, in_queue, done_queue, parse_line_func, exit_event):
        super(Consumer, self).__init__()
        self.in_queue: multiprocessing.Queue = in_queue
        self.done_queue: multiprocessing.Queue = done_queue
        self.parse_line_func: typing.Callable = parse_line_func
        self.exit_event: multiprocessing.Event = exit_event

    def run(self):
        while not self.exit_event.is_set():
            try:
                line = self.in_queue.get(timeout=2)
            except queue.Empty:
                continue
            try:
                sample = self.parse_line_func(line)
                self.done_queue.put(sample)
            except Exception:
                logging.error(traceback.format_exc())
            # finally:
            #     self.in_queue.task_done()
            # AttributeError: 'Queue' object has no attribute 'task_done'
        logging.info(f"{self} 结束退出")


def muti_process_iter_txt(file_path: str,
                          parse_line_func: typing.Callable,
                          read_func: typing.Callable = None,
                          num_max_process=30,
                          verbose=False
                          ):
    in_queue = multiprocessing.Queue(maxsize=10000)
    done_queue = multiprocessing.Queue()
    producer = Producer(in_queue=in_queue, file_path=file_path, read_func=read_func)
    producer.start()
    #
    exit_event = multiprocessing.Event()
    exit_event.clear()
    all_consumer_p = []
    for _ in range(min(os.cpu_count(), num_max_process)):
        p = Consumer(in_queue=in_queue, done_queue=done_queue,
                     exit_event=exit_event,
                     parse_line_func=parse_line_func)
        p.start()
        all_consumer_p.append(p)
    # 进度监控
    start_time = time.time()
    desc_prefix = f"multi_process_read({len(all_consumer_p)}):{'/'.join(file_path.split('/')[-2:])}"
    total_lines = get_num_lines(file_path)
    pbar = tqdm.tqdm(total=total_lines)
    last_finished = cur_total_finished = zero_finished_seconds = 0
    while cur_total_finished < total_lines:
        # in_queue.unfinished_tasks
        try:
            sample = done_queue.get(timeout=5)
            yield sample
        except queue.Empty:
            continue
        except Exception:
            logging.error(traceback.format_exc())
        finally:
            cur_total_finished += 1
        new_finished = cur_total_finished - last_finished
        #
        pbar.set_description(f"{desc_prefix}, finished/total: {cur_total_finished}/{total_lines}")
        pbar.update(new_finished)
        #
        last_finished = cur_total_finished
        if new_finished == 0:
            zero_finished_seconds += 1
            if zero_finished_seconds >= 60:
                _log_ss = f"连续{zero_finished_seconds}s, 没有新任务完成，可能卡住了，主动停止\n{file_path}"
                logging.error(_log_ss)
                raise RuntimeError(_log_ss)
            time.sleep(1)
        else:
            zero_finished_seconds = 0
    #
    if verbose:
        print(f"\n{time.time() - start_time:.2f}s")
    #
    producer.join()
    # in_queue.join() # AttributeError: 'Queue' object has no attribute 'join'
    exit_event.set()
    for p in all_consumer_p:
        p.join()
    # print('---------')
    # while not done_queue.empty():
    #     sample = done_queue.get(timeout=5)
    #     yield sample


#
class FileIterReader(multiprocessing.Process):
    def __init__(self, in_queue, done_queue, parse_line_func, exit_event):
        super(FileIterReader, self).__init__()
        self.in_queue: multiprocessing.Queue = in_queue
        self.done_queue: multiprocessing.Queue = done_queue
        self.parse_line_func: typing.Callable = parse_line_func
        self.exit_event: multiprocessing.Event = exit_event

    def run(self):
        while not self.exit_event.is_set():
            try:
                file_path = self.in_queue.get(timeout=2)
            except queue.Empty:
                continue
            if not os.path.isfile(file_path):
                continue
            for line in open(file_path):
                try:
                    # print(line)
                    sample = self.parse_line_func(line)
                    self.done_queue.put(sample)
                except Exception:
                    logging.error(traceback.format_exc())
            # finally:
            #     self.in_queue.task_done()
            # AttributeError: 'Queue' object has no attribute 'task_done'
        logging.info(f"{self} 结束退出")


def muti_process_iter_txt_v2(file_path: str,
                             parse_line_func: typing.Callable,
                             num_max_process=30,
                             verbose=False
                             ):
    in_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()
    #
    tmp_splited_file_paths = split_file(file_path, out_file_count=num_max_process)
    for _file_path in tmp_splited_file_paths:
        in_queue.put(_file_path)
    #
    exit_event = multiprocessing.Event()
    exit_event.clear()
    all_consumer_p = []
    for _ in range(min(os.cpu_count(), num_max_process)):
        p = FileIterReader(in_queue=in_queue, done_queue=done_queue,
                           exit_event=exit_event,
                           parse_line_func=parse_line_func)
        p.start()
        all_consumer_p.append(p)
    # 进度监控
    start_time = time.time()
    desc_prefix = f"muti_process_iter_txt_v2({len(all_consumer_p)}):{'/'.join(file_path.split('/')[-2:])}"
    total_lines = get_num_lines(file_path)
    pbar = tqdm.tqdm(total=total_lines)
    last_finished = cur_total_finished = zero_finished_seconds = 0
    while cur_total_finished < total_lines:
        # in_queue.unfinished_tasks
        try:
            sample = done_queue.get(timeout=5)
            yield sample
        except queue.Empty:
            continue
        except Exception:
            logging.error(traceback.format_exc())
        finally:
            cur_total_finished += 1
        new_finished = cur_total_finished - last_finished
        #
        pbar.set_description(f"{desc_prefix}, finished/total: {cur_total_finished}/{total_lines}")
        pbar.update(new_finished)
        #
        last_finished = cur_total_finished
        if new_finished == 0:
            zero_finished_seconds += 1
            if zero_finished_seconds >= 60:
                _log_ss = f"连续{zero_finished_seconds}s, 没有新任务完成，可能卡住了，主动停止\n{file_path}"
                logging.error(_log_ss)
                raise RuntimeError(_log_ss)
            time.sleep(1)
        else:
            zero_finished_seconds = 0
    #
    if verbose:
        print(f"\n{time.time() - start_time:.2f}s")
    #
    # in_queue.join() # AttributeError: 'Queue' object has no attribute 'join'
    exit_event.set()
    for p in all_consumer_p:
        p.join()
    # print('---------')
    # while not done_queue.empty():
    #     sample = done_queue.get(timeout=5)
    #     yield sample
    #
    delete(os.path.dirname(tmp_splited_file_paths[0]), verbose=verbose)


# -----

class FileReader(multiprocessing.Process):
    def __init__(self, in_queue, done_queue, read_file_func, exit_event):
        super(FileReader, self).__init__()
        self.in_queue: multiprocessing.Queue = in_queue
        self.done_queue: multiprocessing.Queue = done_queue
        self.read_file_func: typing.Callable = read_file_func
        self.exit_event: multiprocessing.Event = exit_event

    def run(self):
        while not self.exit_event.is_set():
            try:
                file_path = self.in_queue.get(timeout=2)
            except queue.Empty:
                continue
            if not os.path.isfile(file_path):
                continue
            data = self.read_file_func(file_path)
            self.done_queue.put(data)
            # finally:
            #     self.in_queue.task_done()
            # AttributeError: 'Queue' object has no attribute 'task_done'
        logging.info(f"{self} 结束退出")


def muti_process_read_txt_v1(file_path: str,
                             read_file_func: typing.Callable,
                             num_max_process=30,
                             verbose=False
                             ):
    in_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()
    #
    tmp_splited_file_paths = split_file(file_path, out_file_count=num_max_process)
    for _file_path in tmp_splited_file_paths:
        in_queue.put(_file_path)
    #
    exit_event = multiprocessing.Event()
    exit_event.clear()
    all_consumer_p = []
    for _ in range(min(os.cpu_count(), num_max_process)):
        p = FileReader(in_queue=in_queue, done_queue=done_queue,
                       exit_event=exit_event,
                       read_file_func=read_file_func)
        p.start()
        all_consumer_p.append(p)
    # 进度监控
    start_time = time.time()
    desc_prefix = f"muti_process_read_txt({len(all_consumer_p)}):{'/'.join(file_path.split('/')[-2:])}"
    total_samples = len(tmp_splited_file_paths)
    pbar = tqdm.tqdm(total=total_samples)
    last_finished = cur_total_finished = zero_finished_seconds = 0
    while cur_total_finished < total_samples:
        # in_queue.unfinished_tasks
        try:
            sample = done_queue.get(timeout=5)
            yield sample
        except queue.Empty:
            continue
        except Exception:
            logging.error(traceback.format_exc())
        finally:
            cur_total_finished += 1
        new_finished = cur_total_finished - last_finished
        #
        pbar.set_description(f"{desc_prefix}, finished/total: {cur_total_finished}/{total_samples}")
        pbar.update(new_finished)
        #
        last_finished = cur_total_finished
        if new_finished == 0:
            zero_finished_seconds += 1
            if zero_finished_seconds >= 60:
                _log_ss = f"连续{zero_finished_seconds}s, 没有新任务完成，可能卡住了，主动停止\n{file_path}"
                logging.error(_log_ss)
                raise RuntimeError(_log_ss)
            time.sleep(1)
        else:
            zero_finished_seconds = 0
    #
    if verbose:
        print(f"\n{time.time() - start_time:.2f}s")
    #
    # in_queue.join() # AttributeError: 'Queue' object has no attribute 'join'
    exit_event.set()
    for p in all_consumer_p:
        p.join()
    # print('---------')
    # while not done_queue.empty():
    #     sample = done_queue.get(timeout=5)
    #     yield sample
    #
    delete(os.path.dirname(tmp_splited_file_paths[0]), verbose=verbose)


# ------

def muti_process_read_txt(file_path: str,
                          read_file_func: typing.Callable,
                          num_max_process=5,
                          verbose=False
                          ):
    gb_available_mem = (psutil.virtual_memory().available / 1024 / 1024 / 1024)
    n_jobs = min(max(int(gb_available_mem / 25), 2), 10, num_max_process)
    # n_jobs = min(num_max_process, 10)
    tmp_splited_file_paths = split_file(file_path, out_file_count=n_jobs)
    # all_datas = joblib.Parallel(n_jobs=n_jobs,
    #                             # backend='multiprocessing',
    #                             verbose=1)(
    #     joblib.delayed(read_file_func)(_file_path)
    #     for _file_path in tmp_splited_file_paths
    # )
    all_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for _file_path in tmp_splited_file_paths:
            future = executor.submit(read_file_func, _file_path)
            all_futures.append(future)
    all_datas = [f.result() for f in all_futures]
    delete(os.path.dirname(tmp_splited_file_paths[0]), verbose=verbose)
    return all_datas
