import gc
import logging
import threading
import time
import traceback
from functools import wraps


def synchronized(func):
    func.__lock__ = threading.Lock()

    @wraps(func)
    def lock_func(*args, **kwargs):
        with func.__lock__:
            res = func(*args, **kwargs)
        return res

    return lock_func


def async_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        thr = threading.Thread(target=func, args=args, kwargs=kwargs)
        thr.setDaemon(daemonic=True)  # 设为True, 跟随主线程结束
        thr.start()

    return wrapper


# singleton_classes = set()
def __setattr(self, key, value):
    raise PermissionError("constant reassignment error!")


class classproperty(property):
    """
    Usage:
        class Stats:
            _current_instance = None

            @classproperty
            def singleton(cls):
                if cls._current_instance is None:
                    cls._current_instance = Stats()
            return cls._current_instance
    """

    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def singleton(cls):
    """
        只初始化一次；new和init都只调用一次
    """
    instances = {}  # 当类创建完成之后才有内容

    # singleton_classes.add(cls)

    @synchronized
    @wraps(cls)
    def get_instance(*args, **kw):
        if cls not in instances:  # 保证只初始化一次
            # logging.info(f"{cls.__name__} init start ...")
            _instance = cls(*args, **kw)
            cls.__setattr__ = __setattr  # 不可变对象,不可赋值、修改属性
            instances[cls] = _instance
            # logging.info(f"{cls.__name__} init done ...")
            gc.collect()
        return instances[cls]

    return get_instance


class Singleton(object):
    """
        保证只new一次；但会初始化多次，每个子类的init方法都会被调用
        会造成对象虽然是同一个，但因为会不断地调用init方法，对象属性被不断的修改
    """
    instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        """
        :type kwargs: object
        """
        if cls.instance is None:  # 保证只new一次；但会初始化多次，每个子类的init方法都会被调用，造成对象虽然是同一个但会不断地被修改
            cls.instance = super().__new__(cls)
        return cls.instance


def try_catch_with_logging(func=None, default_response=None):
    def out_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except Exception:
                res = default_response
                logging.error(traceback.format_exc())
            return res

    if func is None:
        return out_wrapper  # @try_catch_with_logging()
    return out_wrapper(func)  # @try_catch_with_logging


def retry(func=None, try_times=3, expect_response=None, interval_seconds=5, desc='', verbose=True):
    """  /, *,
    :param func:
    :param / 斜杠位置以前，只能接受位置参数，不能接受关键字参数(func=...); python3.8
    :param * 只接受关键字参数，不接受位置参数
    """

    def out_wrapper(func):
        module_name = func.__module__
        func_name = func.__name__
        prefix = f"{desc}.{module_name}.{func_name}".strip('.')

        @wraps(func)
        def wrapper(*args, **kwargs):
            res = None
            for i in range(1, try_times + 1):
                is_success = False
                try:
                    res = func(*args, **kwargs)
                    is_success = True
                except Exception:
                    logging.error(traceback.format_exc())
                # 判断 response 是否符合预期
                if expect_response is not None:
                    if callable(expect_response):
                        is_success = expect_response(res)
                    else:
                        is_success = (expect_response == res)
                if is_success:
                    break
                #
                if verbose:
                    if i < try_times:
                        print(f"* {prefix}, 第{i}/{try_times}次执行失败, {interval_seconds}s后重试")
                    else:
                        print(f"* {prefix}, 第{i}/{try_times}次执行失败, 不再重试，退出")
                #
                time.sleep(interval_seconds)
            return res

        return wrapper

    if func is None:
        return out_wrapper  # @retry()
    return out_wrapper(func)  # @retry
