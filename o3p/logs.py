import os
import logging
from typing import Optional
import time
import numpy as np
import jax.numpy as jnp

global_first_eval_log_done = None
global_eval_logger = None
global_logger_init_list = None
global_timing_time_saved = None
global_timing_first_time_saved = None


def timing():
    global global_timing_time_saved, global_timing_first_time_saved
    if global_timing_time_saved is None or global_timing_first_time_saved is None:
        global_timing_first_time_saved = time.time()
        global_timing_time_saved = global_timing_first_time_saved
        interval_time = 0.0
        total_time = 0.0
    else:
        interval_time = (time.time() - global_timing_time_saved)
        total_time = time.time() - global_timing_first_time_saved
        global_timing_time_saved = time.time()
    return interval_time, total_time


class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.__level = level

    def filter(self, logrecord):
        return logrecord.levelno == self.__level


def o3p_log_reset():
    global global_first_eval_log_done, global_eval_logger
    if global_eval_logger is not None:
        for handler in global_eval_logger.handlers[:]:
            global_eval_logger.removeHandler(handler)
            handler.close()
    global_first_eval_log_done = None
    global_eval_logger = None
    timing()


def str_conversion(x):
    if isinstance(x, float):
        return f"{x:16.4f}"
    elif isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):
        return f"{np.asarray(np.copy(x), dtype=float):16.4f}"
    elif isinstance(x, int):
        return f"{x:12d}"
    else:
        return str(x).replace(",", ";")


def o3p_log(
    info: dict,
    save_dir: Optional[str] = None,
):
    global global_first_eval_log_done, global_eval_logger, global_logger_init_list
    if global_first_eval_log_done is None:
        if global_eval_logger is not None:
            for handler in global_eval_logger.handlers[:]:
                global_eval_logger.removeHandler(handler)
                handler.close()
        global_first_eval_log_done = True
        if save_dir:
            s_dir = os.path.expanduser(save_dir)
            os.makedirs(s_dir, exist_ok=True)
            print("Logging in " + os.path.join(s_dir, "log.csv"))
            open(os.path.join(s_dir, "log.csv"), "w").close()
        global_eval_logger = logging.getLogger("o3p_log")
        global_eval_logger.setLevel(logging.DEBUG)
        global_eval_logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        chformatter = logging.Formatter("%(message)s")
        ch.setFormatter(chformatter)
        global_eval_logger.addHandler(ch)
        if save_dir:
            fh = logging.FileHandler(
                os.path.join(os.path.expanduser(save_dir), "log.csv")
            )
            fh.setLevel(logging.INFO)
            fhfilter = LevelFilter(logging.INFO)
            fh.addFilter(fhfilter)
            fhformatter = logging.Formatter("%(message)s")
            fh.setFormatter(fhformatter)
            global_eval_logger.addHandler(fh)
        init_list = list(info.keys())
        if "avg_return" in init_list:
            init_list.remove("avg_return")
            init_list = ["avg_return"] + init_list
        if "interval_time" not in init_list:
            init_list = ["interval_time"] + init_list
        else:
            init_list.remove("interval_time")
            init_list = ["interval_time"] + init_list
        if "time" not in init_list:
            init_list = ["time"] + init_list
        else:
            init_list.remove("time")
            init_list = ["time"] + init_list
        if "iterations" in init_list:
            init_list.remove("iterations")
            init_list = ["iterations"] + init_list
        global_logger_init_list = init_list
        global_eval_logger.info(",".join(map(str_conversion, init_list)))
    interval_time, total_time = timing()
    if time not in info:
        info["time"] = total_time
    if interval_time not in info:
        info["interval_time"] = interval_time
    log_info = [info[key] for key in global_logger_init_list]

    # global_eval_logger.warning(str(info))
    if save_dir:
        global_eval_logger.info(",".join(map(str_conversion, log_info)))