from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import seaborn as sb

from typing import Optional, List, Any, Dict, Tuple, Iterable
from scipy.interpolate import interp1d
from scipy.stats import norm, skewnorm


__all__ = ("Statistics",)


class Aggregator:
    @staticmethod
    def bootstrap(size=1000, **kwargs):

        if kwargs.get("samples") is not None:
            samples = kwargs.get("samples")
            idx = np.random.randint(samples.shape[0], size=size)
            return samples[idx]
        loc = kwargs.get("loc")
        scale = np.abs(kwargs.get("scale", 0))
        if "skewness" in kwargs:
            return skewnorm.rvs(kwargs["skewness"], loc=loc, scale=scale, size=size)
        return norm.rvs(loc=loc, scale=scale, size=size)

    @staticmethod
    def reduce(samples, reduction="identity"):
        return getattr(Aggregator, reduction)(samples)

    @staticmethod
    def identity(samples):
        return dict(samples=np.array(samples))

    @staticmethod
    def normal(samples):
        loc, scale = norm.fit(samples)
        return dict(loc=loc, scale=scale)

    @staticmethod
    def skewednormal(samples):
        skewness, loc, scale = skewnorm.fit(samples)
        return dict(skewness=skewness, loc=loc, scale=scale)


class Statistics:
    def __init__(
        self,
        data_dict: Dict[str, int | float | np.ndarray],
        name: str = "stat",
    ):
        self.name = name
        self.__data__ = data_dict

    @classmethod
    def from_samples(
        cls,
        samples: List[float] | np.ndarray,
        reduction: str = "normal",
        name: str = "stat",
    ) -> Statistics:
        num_samples = len(samples)
        reduced = Aggregator.reduce(samples, reduction=reduction)
        if "scale" in reduced:
            reduced["scale"] /= np.sqrt(num_samples)
            reduced["scale"] = np.clip(reduced["scale"], a_min=1e-15, a_max=np.inf)
        return cls(reduced, name=name)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, key: str) -> Statistics:
        subdf = df.filter(regex=f"^{key}")
        data = {}
        for i, col in enumerate(subdf.columns):
            k = col.replace(key, "")
            if not k[0].isalpha():
                k = k[1:]
            data[k] = subdf[col].values
        return cls(data, name=key)

    @property
    def dict(self):
        out = dict()
        for k, v in self.__data__.items():
            out[self.name + "_" + k] = [v] if isinstance(v, float) else v
        return out

    def __repr__(self):
        out = "Statistics("
        for k, v in self.__data__.items():
            out += f"{k}={v}, "

        return out[:-2] + ")"

    def __len__(self):
        return len(self.__data__)

    def __getattr__(self, key):
        if key == "__data__":
            return self.__data__
        return self.__data__[key]

    @property
    def shape(self) -> Tuple[int]:
        d = next(iter(self.__data__.values()))
        return d.shape if isinstance(d, np.ndarray) else ()

    def concat(self, other: Statistics):
        stat_copy = other.copy()
        for k, v in self.__data__.items():
            if not isinstance(v, np.ndarray):
                v = np.array([v])

            if k in other.__data__:
                if not isinstance(other.__data__[k], np.ndarray):
                    stat_copy.__data__[k] = np.array([other.__data__[k]])
                stat_copy.__data__[k] = np.append(other.__data__[k], v)
        stat_copy.__checksizes__()
        return stat_copy

    def __checksizes__(self):
        expected_size = None
        for _, v in self.__data__.items():
            if expected_size is None:
                expected_size = v.shape if isinstance(v, np.ndarray) else 0
            else:
                msg = "Size mismatch!"
                assert expected_size == v.shape if isinstance(v, np.ndarray) else 0, msg

    def copy(self):
        return Statistics(copy.deepcopy(self.__data__))

    def __contains__(self, key):
        return key in self.__data__

    def clip(self, min=None, max=None):
        if "loc" not in self:
            return self.copy()
        min = self["loc"].min() if min is None else min
        max = self["loc"].max() if max is None else max
        other = self.copy()
        other["loc"] = np.clip(self["loc"], min, max)
        return other

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.__data__:
            return self.__data__[key]
        else:
            return Statistics({k: v[key] for k, v in self.__data__.items()})

    def __setitem__(self, key, value):
        if isinstance(key, str) and key in self.__data__:
            self.__data__[key] = value
        else:
            msg = f"Key {key} not found in Statistics {self}."
            raise LookupError(msg)

    def __add__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            if "loc" in data:
                data["loc"] += other.loc
            if "scale" in data:
                data["scale"] += other.scale
            if "samples" in data:
                data["samples"] += other.samples
        else:
            if "loc" in data:
                data["loc"] += other
            if "samples" in data:
                data["samples"] += other
        return Statistics(data)

    def __sub__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            if "loc" in data:
                data["loc"] -= other.loc
            if "scale" in data:
                data["scale"] += other.scale
            if "samples" in data:
                data["samples"] += other.samples
        else:
            if "loc" in data:
                data["loc"] -= other
            if "samples" in data:
                data["samples"] -= other

        return Statistics(data)

    def __mul__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            if "loc" in data:
                data["loc"] *= other.loc
            if "scale" in data:
                data["scale"] = data["loc"] * (self.scale / self.loc + other.scale / other.loc)
            if "samples" in data:
                data["samples"] *= other.samples
        else:
            if "loc" in data:
                data["loc"] *= other
            if "scale" in data:
                data["scale"] *= other
            if "samples" in data:
                data["samples"] *= other
        return Statistics(data)

    def __truediv__(self, other):
        data = self.copy().__data__

        if not isinstance(other, Statistics):
            if "loc" in data:
                data["loc"] /= other
            if "scale" in data:
                data["scale"] /= other
            if "samples" in data:
                data["samples"] /= other
        else:
            self_copy = self.copy().__data__
            other_copy = other.copy().__data__
            if "loc" in data:
                data["loc"] /= other.loc

            if "scale" in data:
                if isinstance(self_copy["scale"], np.ndarray):
                    if "loc" in self_copy:
                        self_copy["loc"][self.loc == 0] = 1
                        other_copy["loc"][other.loc == 0] = 1
                    if "scale" in self_copy:
                        self_copy["scale"][self.loc == 0] = 0
                        other_copy["scale"][other.loc == 0] = 0

                elif isinstance(self_copy["scale"], (int, float)):
                    if "loc" in data:
                        self_copy["loc"] = 1 if self.loc == 0 else self.loc
                        other_copy["loc"] = 1 if other.loc == 0 else other.loc
                    if "scale" in data:
                        self_copy["scale"] = 0 if self.loc == 0 else self.scale
                        other_copy["scale"] = 0 if self.loc == 0 else other.scale

                data["scale"] = np.abs(data["loc"] * (self.scale / self.loc - other.scale / other.loc))
            if "samples" in data:
                data["samples"] /= other.samples
        return Statistics(data)

    def __ge__(self, other):
        return self["loc"] >= other.loc

    def __gt__(self, other):
        return self["loc"] > other["loc"]

    def __le__(self, other):
        return self["loc"] <= other["loc"]

    def __lt__(self, other):
        return self["loc"] < other["loc"]

    def __eq__(self, other):
        return self["loc"] == other["loc"]

    def rescale_(self, s):
        if "scale" in self:
            self["scale"] /= s
        return

    def bootstrap(self, indexes: Optional[Iterable[bool]] = None, size=1000):
        if "samples" in self:
            idx = np.random.randint(len(self.samples), size=size)
            return self.samples[idx]
        if isinstance(next(iter(self.__data__.values())), np.ndarray):
            n = len(next(iter(self.__data__.values())))
            samples = np.zeros((n, size))

            for i in range(n):
                if indexes is not None and not indexes[i]:
                    continue
                params = {k: v[i] for k, v in self.__data__.items()}
                samples[i] = Aggregator.bootstrap(size=size, **params)
            return samples

        return Aggregator.bootstrap(size=size, **self.__data__)

    def interpolate(
        self,
        x: pd.Series | np.ndarray | List[Any],
        aux: Optional[pd.Series | np.ndarray | List[Any]] = None,
        size=100,
        **kwargs,
    ):
        if not isinstance(x, pd.Series):
            x = pd.Series(x, name="xaxis")
        if aux is not None and not isinstance(aux, pd.Series):
            aux = pd.Series(aux, name="aux")
        elif aux is None:
            aux = pd.Series([None] * len(x), name="aux")
        yinterp = None
        xinterp = None
        auxinterp = None
        kwargs.setdefault("kind", "cubic")
        for _aux in aux.unique():
            idx = aux == _aux
            _x = x if _aux is None else x[idx]
            aug_x = np.linspace(_x.min(), _x.max(), size)
            aug_y = dict()
            for k, v in self.__data__.items():
                _v = v if _aux is None else v[idx]
                aug_y[k] = interp1d(_x.values, _v, **kwargs)(aug_x)

            # aug_y = interp1d(_x.values, _y, **kwargs)(aug_x)
            if xinterp is None:
                xinterp = pd.Series(aug_x, name=x.name)
                auxinterp = pd.Series([_aux] * len(xinterp), name=aux.name)
                yinterp = Statistics(aug_y)
            else:
                xinterp = pd.concat(
                    [xinterp, pd.Series(aug_x, name=x.name)],
                    ignore_index=True,
                )
                auxinterp = pd.concat(
                    [
                        auxinterp,
                        pd.Series([_aux] * len(xinterp), name=aux.name),
                    ],
                    ignore_index=True,
                )
                yinterp = self.concat(Statistics(aug_y))

            if _aux is None:
                auxinterp = None
                return xinterp, yinterp

        return xinterp, yinterp, auxinterp

    def lineplot(
        self,
        x: pd.Series | np.ndarray | List[Any],
        aux: Optional[pd.Series | np.ndarray | List[Any]] = None,
        interpolate: Optional[str] = None,
        indexes: Optional[Iterable[bool]] = None,
        n_boot: int = 100,
        **kwargs,
    ):
        if interpolate is None:
            bs = self.bootstrap(indexes, n_boot)
            df = pd.DataFrame.from_records(bs)
        else:
            out = self.interpolate(x, aux=aux, kind=interpolate)
            if aux is None:
                x, y = out
            else:
                x, y, aux = out
            bs = y.bootstrap(indexes, n_boot)
            df = pd.DataFrame.from_records(bs)
        if not isinstance(x, pd.Series):
            x = pd.Series(x, name="xaxis")
        if aux is not None and not isinstance(aux, pd.Series):
            aux = pd.Series(aux, name="aux")
        if indexes is not None:
            x = x.loc[indexes]
            aux = aux.loc[indexes] if aux is not None else None
        df[x.name] = x
        if aux is not None:
            df[aux.name] = aux

        df = df.melt(
            [x.name, aux.name] if aux is not None else x.name,
            value_name=self.name,
        )

        return sb.lineplot(
            df,
            y=self.name,
            x=x.name,
            hue=aux.name if aux is not None else None,
            **kwargs,
        )
