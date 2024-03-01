from __future__ import annotations

from itertools import product
from typing import Any, Generator, List, Type, Optional
from copy import deepcopy
import functools
import pickle
import json


def static(cls):
    @functools.wraps(cls, updated=())
    class StaticConfig(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            self.lock_types()

    return StaticConfig


def frozen(cls):
    @functools.wraps(cls, updated=())
    class FrozenConfig(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            self.lock()

    return FrozenConfig


class Config:
    separator: str = "."

    def __init__(self, name: str, as_seq: bool = True, **kwargs: Any):

        self._state: dict = {}
        self._name: dict = name
        self._type_lock: bool = False
        self._lock: bool = False
        self.__types__: dict[str, Type] = {}
        self.__sequence_params__: list[str] = []
        self._as_seq: bool = as_seq

        self.name: str = name

        for key in kwargs:
            if key == "dict":
                raise KeyError("Key dict is a protected attribute.")
            if key == "keys":
                raise KeyError("Key keys is a protected attribute.")
            setattr(self, key, kwargs[key])

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]

        return self._state.get(name, None)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__ or key in self.dict

    def __setattr__(self, name: str, value: Any) -> None:
        if name in [
            "_name",
            "_state",
            "_lock",
            "_type_lock",
            "_as_seq",
            "__types__",
            "__sequence_params__",
        ]:
            object.__setattr__(self, name, value)
        else:
            if self._lock:
                raise Exception("Config is locked.")
            if self._type_lock and name not in self._state:
                raise Exception(f"In type-lock mode, new param `{name}` cannot be added.")
            if self._type_lock and not self.is_type(value, self.type(name)):
                raise Exception(f"In type-lock mode, `{name}` must be of type `{self.type(name).__name__}`.")
            if isinstance(value, list) and not self.is_one_type(value):
                raise ValueError(
                    f"Value is `list` {name} but contains multiple types: `{[type(v).__name__ for v in value]}`."
                )
            self._state[name] = value
            self.__types__[name] = type(value) if not isinstance(value, list) else type(value[0])
            if isinstance(value, list):
                self.__types__[name] = type(value[0])
                if self._as_seq:
                    self.__sequence_params__.append(name)
            else:
                self.__types__[name] = type(value)
                if name in self.__sequence_params__:
                    self.__sequence_params__.remove(name)

    def __getstate__(self):
        return self._state

    def __setstate__(self, d):
        if self._lock == False:
            self._state.update(d)
        else:
            raise Exception("Config is locked.")

    def __getitem__(self, key):
        if key in self._state:
            return self._state[key]
        return None

    def __dir__(self):
        return self._state.keys()

    def __setitem__(self, key, value):
        if self._lock == False:
            self._state[key] = value
        else:
            raise Exception("Config is locked.")

    def __iter__(self):
        return self._state.__iter__()

    def __len__(self) -> int:
        return len(list(self.to_sequence()))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return json.dumps(self.dict, indent=2, default=str)

    def save(self, path):
        json.dump(self.dict, open(path, "w"), default=str)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.load(open(path, "r")))

    def lock(self):
        """Freeze the state of the config."""
        self._lock = True
        return self

    def unlock(self):
        """Unfreeze the state of the config."""
        self._lock = False
        return self

    def lock_types(self):
        """Lock the types of the config."""
        self._type_lock = True
        return self

    def unlock_types(self):
        """Unlock the types of the config."""
        self._type_lock = False
        return self

    @staticmethod
    def _convert_json_res(res: dict):
        """Convert the download config, meta into a Config."""
        meta = {}
        config = Config()

        if res is None:
            return Config(), {}

        for key, item in res.get("__meta__", {}).items():
            meta[key] = item

        config = Config.from_dict(res.get("__config__", {}))
        return config, meta

    def keys(self):
        """Return the keys of the config."""
        return self._state.keys()

    def values(self):
        """Return the values of the config."""
        return self._state.values()

    def items(self):
        """Return the items of the config."""
        return self._state.items()

    def get(self, name, default: Optional[Any] = None):
        """Return the value of the parameter."""
        return self._state.get(name, default)

    def pop(self, name, default: Optional[Any] = None):
        """Pop the value of the parameter."""
        val = self._state.pop(name, default)
        self = Config(**self._state)
        # self.__types__.pop(name, None)
        return val

    def type(self, name: str) -> Type:
        """Return the type of the parameter."""
        return self.__types__[name]

    @property
    def name(self):
        return self._name

    @staticmethod
    def is_type(value: Any, expected_type: Type) -> bool:
        """Check if the value is of the expected type. If the value is a list, then check if all elements are of the expected type."""
        return isinstance(value, expected_type) or (
            isinstance(value, list) and all([isinstance(v, expected_type) for v in value])
        )

    @staticmethod
    def is_one_type(value: bool) -> bool:
        """Check if the value is a list and contains only one type."""
        if not issubclass(value.__class__, list) or len(value) == 0:
            return True
        t = type(value[0])
        return all([isinstance(v, t) for v in value])

    def update(self, **kwargs: Any):
        """Update the config with new parameters."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of a parameter.

        Args:
            key (str): The key of the parameter.
            default (Any, optional): The default value to return if the key is not found. Defaults to None.

        Example:
        -------------
        config = Config(a=4)
        assert config.get("a") == 4

        """
        components = key.split(self.separator)
        if (key not in self or self._state[key] is None) and len(components) == 1:
            return default
        if key in self and self._state[key] is not None:
            return self._state[key]

        subkey = self.separator.join(components[1:])
        assert not isinstance(
            self._state[components[0]], (list, tuple, set)
        ), f"In get: Component {components[0]} of key {key} must not be iterable."
        return self._state[components[0]].get(subkey, default)

    @property
    def dict(self) -> dict[str, Any]:
        """A property that converts a config to a dict. Supports nested Config."""
        d = {}
        for key, item in self._state.items():
            if issubclass(item.__class__, Config):
                d[key] = item.dict
            elif issubclass(item.__class__, list):
                d[key] = [i.dict if issubclass(i.__class__, Config) else i for i in item]
            else:
                d[key] = item
        return d

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        as_seq: bool = True,
    ) -> Config:
        """Convert a dict to a Config object.
        `data` can be nested (dict within dict), which will generate sub configs.
        If `data` is not a dict, then the method returns `data`.

        Example:
        -------------
        d = {"a": {"b": 4}}
        config = Config.from_dict(d)
        assert config.a.b == 4
        """
        if isinstance(data, dict) == False:
            return data

        config = Config(data["name"], as_seq=as_seq)
        for key in data:
            if isinstance(data[key], dict):
                config[key] = Config.from_dict(data[key])
            elif issubclass(data[key].__class__, list) and isinstance(data[key][0], dict):
                config[key] = [Config.from_dict(d) for d in data[key]]
            else:
                # config[key] = data[key]
                setattr(config, key, data[key])
            if isinstance(config[key], list):
                config.as_sequence(key)
        config.lock()
        config.lock_types()
        return config

    @classmethod
    def auto(
        cls,
        config: Optional[str | Config | List[Config]],
        *args: Any,
        **kwargs: Any,
    ) -> Config | list[Config]:
        """
        Automatically build a Config object from a string, Config object, or list of Config objects.

        Args:
            config (str, Config, list[Config]): The configuration to build.
            *args (Any): The arguments to pass to the Config object.
            **kwargs (Any): The keyword arguments to pass to the Config object.
        """
        if config is None:
            return
        configs = [config] if not isinstance(config, list) else config
        res = []
        for config in configs:
            if config in dir(cls):
                res.append(getattr(cls, config)(*args, **kwargs))
            elif isinstance(config, cls):
                res.append(config)
            else:
                t = config if isinstance(config, str) else type(config)
                message = f"Invalid config type `{t}` for auto build of object `{cls.__name__}`."
                raise TypeError(message)
        if len(res) == 1:
            return res[0]
        elif len(res) == 0:
            return
        return res

    def is_sequenced(self) -> bool:
        """Check if the config is sequenced."""
        for k, v in self._state.items():
            if self.is_sequence(k):
                return True
            elif isinstance(v, Config) and v.is_sequenced():
                return True
        return False

    def as_sequence(self, key: str) -> None:
        """Set a parameter as a sequence."""
        if self.is_sequence(key):
            return
        if isinstance(self._state[key], (list, tuple, set)):
            self.__sequence_params__.append(key)

    def not_sequence(self, key: str) -> None:
        """Set a parameter as not a sequence."""
        if not self.is_sequence(key):
            return
        if key in self.__sequence_params__:
            self.__sequence_params__.remove(key)

    def is_sequence(self, key: str) -> bool:
        """Check if a parameter is a sequence."""
        return key in self.__sequence_params__

    def to_sequence(self) -> Generator[Config]:
        """Convert the config to a sequence of configs."""
        if not self.is_sequenced():
            yield self
            return

        keys_to_scan = []
        values_to_scan = []

        for k, v in self._state.items():
            if self.is_sequence(k):
                keys_to_scan.append(k)
                values_to_scan.append(v)
            elif issubclass(v.__class__, Config) and v.is_sequenced():
                keys_to_scan.append(k)
                values_to_scan.append(v.to_sequence())
        for values in product(*values_to_scan):
            config = self.copy()
            config.unlock()
            for k, v in zip(keys_to_scan, values):
                setattr(config, k, v)
            if config.is_sequenced():
                for c in config.to_sequence():
                    ext = self.extension(c)
                    c.name = self.name + (ext if ext != "" else "")
                    yield Config(**c)
            else:
                ext = self.extension(config)
                config.name = self.name + (ext if ext != "" else "")
                yield Config(**config)

    def copy(self):
        """Return a copy of the config."""
        return self.__class__(**self._state.copy())

    def extension(self, config: Config):
        """Return the extension of the config containing sequenced values."""
        ext = ""
        for k, v in self._state.items():
            if isinstance(v, list) and all([issubclass(vv.__class__, Config) for vv in v]):
                ext += Config.separator + config[k].name
            elif issubclass(v.__class__, Config):
                ext += self[k].extension(config[k])
        return ext

    def summarize_subconfig(self, config: Config):
        """Summarize the subconfig using the values that are sequenced."""
        values = {}
        for k, v in self._state.items():
            if self.is_sequence(k) and not issubclass(v[0].__class__, Config):
                values[k] = config._state[k]
            elif self.is_sequence(k) and issubclass(v[0].__class__, Config):
                for vv in v:
                    if vv.name != config._state[k].name:
                        continue
                    values.update(
                        {k + self.separator + _k: _v for _k, _v in vv.summarize_subconfig(config._state[k]).items()}
                    )
                    break
            elif issubclass(v.__class__, Config):
                values.update(
                    {k + self.separator + _k: _v for _k, _v in v.summarize_subconfig(config._state[k]).items()}
                )
        return values
