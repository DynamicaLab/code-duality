import psutil
import numpy as np
import os

from datetime import datetime, timedelta
from typing import Any


class MetricsCallback:
    @classmethod
    def to_setup(cls, *args, **kwargs):
        obj = cls(*args)
        obj.setup(**kwargs)
        return obj

    def setup(self, **kwargs: Any) -> None:
        pass

    def update(self) -> None:
        raise NotImplementedError()

    def teardown(self, **kwargs: Any) -> None:
        pass


class Progress(MetricsCallback):
    def setup(self, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.logger = kwargs.get("logger")
        self.counter = 0
        self.begin = datetime.now()
        self.last = datetime.now()
        self.now = datetime.now()
        self.from_start = self.now - self.begin

    @staticmethod
    def format_dt(dt):
        idays, isec = dt.days, dt.seconds
        imsec = dt.microseconds
        return "{}-{}.{}".format(idays, timedelta(seconds=isec), int(100 * round(imsec / 1e6, 2)))

    def update(self) -> None:
        self.counter += 1
        self.now = datetime.now()
        self.from_start = self.now - self.begin
        self.from_last = self.now - self.last
        self.remaining = timedelta(seconds=self.from_start.total_seconds() * ((self.total / self.counter) - 1))
        msg = f"Progress: "
        msg += f"{self.counter} / {self.total}" if self.total is not None else f"{self.counter}"
        msg += f"    elapsed {self.format_dt(self.from_start)},"
        msg += f"    remaining {self.format_dt(self.remaining)},"
        msg += f"    per step {self.format_dt(self.from_last)},"
        if self.logger is not None:
            self.logger.info(msg)
        self.last = self.now

    def teardown(self, **kwargs: Any) -> None:
        msg = f"Progress:    total {self.format_dt(self.from_start)}"
        if self.logger is not None:
            self.logger.info(msg)


class MemoryCheck(MetricsCallback):
    def __init__(self, unit: str = "gb"):
        pw = {"b": 0, "kb": 1, "mb": 2, "gb": 3}
        if unit not in pw.keys():
            raise ValueError(f"`{unit}` is an invalid unit, valid units are" + "`[b, kb, mb, gb]`.")
        self.factor = 1024 ** pw[unit.lower()]
        self.unit = unit.upper()

    def setup(self, **kwargs) -> None:
        self.logger = kwargs.get("logger")
        self.memory_usage = [round(psutil.virtual_memory().used / self.factor, 4)]

    def update(self) -> None:
        self.memory_usage.append(round(psutil.virtual_memory().used / self.factor, 4))
        msg = f"Memory:    {np.mean(self.memory_usage): .4f}(+- {np.std(self.memory_usage):.4f}) {self.unit}"
        if self.logger is not None:
            self.logger.info(msg)

    def teardown(self, **kwargs: Any) -> None:
        msg = f"Memory:    max usage {max(self.memory_usage)} {self.unit}"
        if self.logger is not None:
            self.logger.info(msg)


class Checkpoint(MetricsCallback):
    def setup(self, **kwargs: Any) -> None:
        self.patience = kwargs.get("patience", 1)
        self.savepath = kwargs.get("savepath", "./")
        self.logger = kwargs.get("logger")
        self.metrics = kwargs.get("metrics")
        assert self.metrics is not None, "Checkpoint must set up with metrics."
        self.metrics.read_pickle(self.savepath)
        self.counter = 0

    def update(self) -> None:
        if self.counter % self.patience == 0:
            msg = f"Checkpoint: saved at {self.savepath}."
            self.logger.info(msg)
            self.metrics.to_pickle(self.savepath)
        self.counter += 1
