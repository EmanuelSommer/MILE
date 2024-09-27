"""Module for I/O and validation of the configuration."""
import dataclasses
import enum
import importlib.util
import inspect
import itertools
import json
import operator
import random
import sys
import typing
from functools import reduce
from json.encoder import JSONEncoder
from pathlib import Path
from types import GenericAlias, UnionType

YAML_AVAILABLE = importlib.util.find_spec('yaml')
if YAML_AVAILABLE:
    import yaml

YAML_SUFFIXES = ('.yaml', '.yml')
JSON_SUFFIXES = ('.json',)


JSON_TYPES = (int, float, str, bool, list, dict, type(None))

if sys.version_info >= (3, 10):
    JsonPrimitive: typing.TypeAlias = int | float | str | bool | None
    JsonArrayType: typing.TypeAlias = list[
        typing.Union[JsonPrimitive, 'JsonObjectType', 'JsonArrayType']
    ]
    JsonObjectType: typing.TypeAlias = dict[
        str, typing.Union[JsonPrimitive, JsonArrayType, 'JsonObjectType']
    ]
    JsonType: typing.TypeAlias = JsonPrimitive | JsonArrayType | JsonObjectType

    JsonSerializableDict: typing.TypeAlias = dict[str, JsonType]

else:
    JsonPrimitive: typing.TypeAlias = typing.Union[int, float, str, bool, None]
    JsonArrayType: typing.TypeAlias = typing.List[
        typing.Union[JsonPrimitive, 'JsonObjectType', 'JsonArrayType']
    ]
    JsonObjectType: typing.TypeAlias = typing.Dict[
        str, typing.Union[JsonPrimitive, JsonArrayType, 'JsonObjectType']
    ]
    JsonType: typing.TypeAlias = typing.Union[
        JsonPrimitive, JsonArrayType, JsonObjectType
    ]
    JsonSerializableDict: typing.TypeAlias = typing.Dict[str, JsonType]

# Types for Defining tree structured definitions for Grid or Random Generation of
# Configs used for example in Hyperparameter Search in Optimization context.

# Constraint: Leafes must be iterables of json serializable types.
SearchLeafType: typing.TypeAlias = list[JsonType]
SearchTreeType: typing.TypeAlias = dict[
    str, typing.Union[SearchLeafType, 'SearchTreeType']
]


class ConfigJSONEncoder(JSONEncoder):
    """
    Custom JSON Encoder Allowing Serialization for Enum and Custom Non-JSON Objects.

    Notes
    ----------------
    - The Encoder will try to serialize the object using a`config_to_json` method
    if available.
    - If the object is not JSON serializable, it will raise a TypeError.
    """

    def default(self, obj):  # noqa: D102
        if isinstance(obj, enum.Enum):  # Serialize Enum Objects
            return obj.value
        else:
            if getattr(obj, 'config_to_json', None) and callable(obj.config_to_json):
                json_str = obj.config_to_json()
                if not isinstance(json_str, str):
                    raise TypeError(
                        '`config_to_json` method must return a json string, '
                        f'got {type(json_str).__name__}'
                    )
                _ = json.loads(json_str)  # Check if the string is a valid JSON
            else:
                raise TypeError(
                    f'Object of type {obj.__class__.__name__} is not JSON serializable.'
                    f' You can implement a `.config_to_json()` method for '
                    f'{obj.__class__.__name__} to make it JSON serializable. '
                    'But note that in loading the configuration, customly '
                    'serialized objects should be deserialized back to the '
                    'original object by extending the `__post_init__` method '
                    'of the BaseConfig class. This is usually not recommended.'
                )
        return json_str


def _serialize(obj):
    """Try to serialize the object using the custom JSON Encoder."""
    return json.dumps(obj, cls=ConfigJSONEncoder).strip('"')


class CustomEnumMeta(enum.EnumMeta):
    """Custom Enum Meta Class for Better Error Handling."""

    def __call__(cls, value, **kwargs):
        """Extend the __call__ method to raise a ValueError."""
        if value not in cls._value2member_map_:
            raise ValueError(
                f'{cls.__name__} must be one of {[*cls._value2member_map_.keys()]}'
            )
        return super().__call__(value, **kwargs)


class BaseEnum(enum.Enum, metaclass=CustomEnumMeta):
    """BaseEnum Class for implementing custom Enum classes."""


if sys.version_info >= (3, 11):

    class BaseStrEnum(enum.StrEnum, metaclass=CustomEnumMeta):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value

    class BaseIntEnum(enum.IntEnum, metaclass=CustomEnumMeta):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value.__str__()

else:

    class BaseStrEnum(str, BaseEnum):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value

    class BaseIntEnum(int, BaseEnum):
        """BaseEnum Class for implementing custom Enum classes."""

        def __str__(self):
            """Return the string representation of the Enum."""
            return self.value.__str__()


C = typing.TypeVar('C', bound='BaseConfig')


@dataclasses.dataclass(frozen=True)
class BaseConfig:
    """Base Configuration Class.

    Functionality
    ----------------
    Base Configuration Class for all the configuration classes.
    It builds on top of the dataclass module to provide additional
    functionality. Such as loading and saving configurations from
    YAML and JSON files, generating schemas and validating
    the attributes against pre-defined type hints.

    Notes
    ----------------
    - The configuration class should be defined using the dataclass decorator.
    - Prefer using 'BaseEnum' 'BaseStrEnum' and 'BaseIntEnum' for Enum fields.
    - The configuration class is frozen to ensure immutability but it can be
    modified using the 'replace' method or by _modify_field method in-place
    the latter is not recommended.
    - The configuration class should be type hinted for all the attributes.
    - Field types must be json serializable to ensure proper Save/Load functionality
    """

    def __post_init__(self):
        """Run the post initialization checks and modifications.

        In the BaseConfig implementation it checks the type hints and
        handles the Enum fields it can be extended in the child classes
        as follows::

            def __post_init__(self):
                super().__post_init__()
                # Additional checks and modifications
                assert 1 < 2
        """
        for field in dataclasses.fields(self):
            if inspect.isclass(field.type):
                if issubclass(field.type, enum.Enum):  # Parse to Enum Types
                    self._modify_field(
                        **{field.name: field.type(getattr(self, field.name))}
                    )

            if not check_type(
                getattr(self, field.name), field.type
            ):  # Check Type Hints
                raise TypeError(
                    '\n'
                    f'| loc: {self.__class__.__name__}.{field.name}\n'
                    f'| expects: {field.type}\n'
                    f'| got: {type(getattr(self, field.name))}\n'
                    f'| description: {field.metadata.get("description")}\n'
                )

    def __contains__(self, item):
        """Check if the item is in the configuration."""
        return item in self.to_dict()

    def __getitem__(self, item: str):
        """Get the item from the configuration."""
        return getattr(self, item)

    def __str__(self):
        """Serialize Configuration to a json string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=3)

    def get_string_paths(self):
        """Get the paths of the configuration."""
        return get_all_string_path(self.to_dict())

    def get_by_path(self, path: list[str] | str) -> JsonType:
        """Get the value in the configuration by path.

        Parameters
        -----------
        path: list[str] | str
            Path to the value in the configuration, either as a list of strings
            or a string with '.' separated sequence of keys.
        """
        if isinstance(path, str):
            path = path.split('.')
        return get_by_path(self, path)

    @classmethod
    def field_names(cls) -> tuple[str]:
        """Get the field names of the configuration."""
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def get_all_subclasses(cls: type[C]) -> set[type[C]]:
        """Get all subclasses of a class."""
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in c.get_all_subclasses()]
        )

    @classmethod
    def config_validate(cls: type[C], obj: str | dict | C) -> C:
        """
        Validate the object against the configuration class.

        Parameters
        ----------------
        obj: Any
            Object to be validated.
        """
        if isinstance(obj, cls):
            return cls.from_dict(obj.to_dict())
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        if isinstance(obj, str):
            return cls.from_dict(json.loads(obj))
        raise ValueError(
            f'Object must be an instance of {cls.__name__}, dict, or json string, '
            f'got {type(obj).__name__}'
        )

    def _modify_field(self, /, **changes):
        """Replace the fields of the configuration in place. (Insecure)."""
        for name, value in changes.items():
            if name not in self.field_names():
                raise ValueError(
                    f'Invalid field name: `{name}` in `{self.__class__.__name__}`'
                )
            object.__setattr__(self, name, value)

    def replace(self, /, **changes):
        """Replace the fields and return a new object with replaced values (Secure).

        Example::

         @dataclass(frozen=True)
         class SomeConfig(BaseConfig):
            x: int
            y: int

         c = SomeConfig(1, 2)
         c_new = c.replace(x=3)
         assert c_new.x == 3 and c_new.y == 2
        """
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_yaml(cls, path: Path | str):
        """Load configuration from a YAML file.

        Parameters
        ----------------
        path: Path | str
            Path to the YAML file.
        """
        return cls.from_dict(cls._yaml_load(path))

    def to_yaml(self, path: Path | str):
        """Deserialize the configuration to a YAML file.

        Parameters
        ----------------
        path: Path | str
            Path to the file to write the configuration.
        """
        return self._yaml_dump(self.to_dict(), path)

    @classmethod
    def from_json(cls, path: Path | str):
        """Deserialize the configuration from a JSON file.

        Parameters
        ----------------
        path: Path | str
            Path to the JSON file.
        """
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Path | str):
        """Serilize the configuration to a JSON file.

        Parameters
        ----------------
        path: Path | str
            Path to the file to write the configuration.
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=3)

    @classmethod
    def schema_to_json(cls, path: Path | str):
        """Dump the JSON schema for the configuration.

        Parameters
        ----------------
        path: Path | str
            Path to the file to write the `json` schema.

        Notes
        ----------------
        - This method is useful for generating configuration templates.
         As it writes a schema for the configuration with type hints,
         desciptions and default values (if available) for each field.
         The user only needs to fill in the values in the schema
         and load the config using either `from_yaml` or `from_json` classmethods.
        """
        with open(path, 'w') as f:
            json.dump(cls.to_schema(), f, indent=3)

    @classmethod
    def schema_to_yaml(cls, path: Path | str):
        """Dump the YAML schema for the configuration.

        Parameters
        ----------------
        path: Path | str
            Path to the file to write the `yaml` schema.

        Notes
        ----------------
        - This method is useful for generating configuration templates.
         As it writes a schema for the configuration with type hints,
         desciptions and default values (if available) for each field.
         The user only needs to fill in the values in the schema
         and load the config using either `from_yaml` or `from_json` classmethods.
        """
        cls._yaml_dump(cls.to_schema(), path)

    @classmethod
    def from_dict(cls, config: dict[str, typing.Any], allow_extra: bool = False):
        """Parse the configuration from a python dictionary.

        Parameters
        ----------------
        config: dict[str, typing.Any]
            Configuration dictionary to be parsed.
        allow_extra: bool
            Whether to allow extra fields in the configuration.

        Notes
        ----------------
        - Configuration does not allow extra fields, meaning that the
        dictionary should only contain the field names defined in the configuration.
        """
        if not allow_extra:
            _check_extra_names(config, {f.name for f in dataclasses.fields(cls)})
        return cls(
            **{
                f.name: cls._handle_field(f, config.get(f.name))
                for f in dataclasses.fields(cls)
                if f.name in config
            }
        )

    def to_dict(self) -> JsonSerializableDict:
        """Convert the configuration to json serializable python dictionary.

        Notes
        ----------------
        The method will recursively convert the configuration models to a

        """
        return {
            f.name: (
                f.type.to_dict(getattr(self, f.name))
                if inspect.isclass(f.type) and issubclass(f.type, BaseConfig)
                else (
                    getattr(self, f.name)
                    if type(getattr(self, f.name)) in JSON_TYPES
                    else _serialize(getattr(self, f.name))
                )
            )
            for f in dataclasses.fields(self)
        }

    @classmethod
    def from_dir(cls, path: Path | str):
        """Load all configuration from a selected directory.

        Parameters
        ----------------
        path: Path | str
            Path to the directory containing configuration files.

        Returns
        ----------------
        list[BaseConfig]:
            List of configuration objects.

        Notes
        ----------------
        - In the directory `yaml` and `json` files can be mixed, the method will
        load all of them.
        """
        if not isinstance(path, Path):
            path = Path(path)
        patt = f'*[{"|".join(YAML_SUFFIXES + JSON_SUFFIXES)}]'
        return [cls.from_file(p) for p in path.glob(patt)]

    @classmethod
    def from_file(cls, path: Path | str):
        """Load configuration from a file.

        Parameters
        ----------------
        path: Path | str
            Path to the configuration file.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if path.suffix in YAML_SUFFIXES:
            return cls.from_yaml(path)
        if path.suffix in JSON_SUFFIXES:
            return cls.from_json(path)
        raise ValueError(
            f'Unsupported file format: {path.suffix}. '
            f'Please use one of {YAML_SUFFIXES + JSON_SUFFIXES}'
        )

    @staticmethod
    def _handle_field(f: dataclasses.Field, value):
        """Handle the field value.

        Notes
        ------
        - Internal method used in parsing the configuration from a dictionary.
        """
        if inspect.isclass(f.type):
            if issubclass(f.type, BaseConfig):
                return f.type.from_dict(value if value else {})
        return value

    @classmethod
    def to_schema(cls):
        """Generate a dictionary schema for the configuration.

        Notes
        ----------------
        - Schema is a python dictionary with field names as keys and as values
        we have either a default value (if available) a type hint with
        a description (if available).
        - For Enum fields, the schema will display the possible values
        as a string of set of possible values.
        """
        return {
            f.name: (
                f.type.to_schema()
                if inspect.isclass(f.type) and issubclass(f.type, BaseConfig)
                else (
                    (
                        f.default
                        if not isinstance(f.default, dataclasses._MISSING_TYPE)
                        else (
                            f'{f.type.__name__}: '
                            f"{f.metadata.get('description', 'No description given')}"
                        )
                    )
                    if not isinstance(f.type, enum.EnumMeta)
                    else str(set(f.type._value2member_map_.keys())).replace("'", '')
                )
            )
            for f in dataclasses.fields(cls)
        }

    @staticmethod
    def _yaml_dump(obj, path: Path | str):
        """Dump the object to a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                'Please install "pyyaml" module to use the YAML serialization, '
                'otherwise use .to_json() method.'
            )
        with open(path, 'w') as f:
            yaml.dump(obj, f, sort_keys=False)

    @staticmethod
    def _yaml_load(path: Path | str):
        """Load the object from a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError(
                'Please install "pyyaml" module to use the YAML deserialization, '
                'otherwise use .from_json() method.'
            )
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def to_search_tree(self, prune_null: bool = True) -> SearchTreeType:
        """Generate a search tree template for the configuration class instance.

        Parameters
        ----------------
        prune_null: bool
            Whether to prune the null paths from the search tree.

        Returns
        ----------------
        SearchTreeType:
            A tree structure of the configuration class where the leafes
            are List of possible values to use in Grid Search or Random Search.

        Notes
        ----------------
         - Configurations which are not searchable are not included in the tree.
         - We mark the searchable fields with a metadata key 'searchable'.
        """
        search_tree = _to_search_tree(self)
        if prune_null:
            search_tree = prune_null_path(search_tree)
        return search_tree

    def search_tree_to_yaml(self, path: Path | str):
        """Dump the Search Tree Template to a YAML file."""
        self._yaml_dump(self.to_search_tree(), path)

    def search_tree_to_json(self, path: Path | str):
        """Dump the Search Tree Template to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_search_tree(), f, indent=3)

    def get_configs_grid(self, search_tree: SearchTreeType):
        """Generate a grid of configurations from the Search Tree.

        Parameters
        ----------------
        search_tree: SearchTreeType
            Search Tree that defines the search space.

        Returns
        ----------------
        list[BaseConfig]:
            List of configurations generated from the Search Tree in Grid Search.
        """
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(), search_tree=search_tree
            )
        ]

    def yield_configs_grid(self, search_tree: SearchTreeType):
        """Yield a grid of configurations one-by-one from the Search Tree.

        Parameters
        ----------------
        search_tree: SearchTreeType
            Search Tree that defines the search space.

        Returns
        ----------------
        Generator[BaseConfig]:
            Generator of configurations generated from the Search Tree in Grid Search.
        """
        for cfg in _yield_config_search_space(
            config_tree=self.to_dict(), search_tree=search_tree
        ):
            yield self.from_dict(cfg)

    def get_configs_random(self, search_tree: SearchTreeType, n: int, seed: int):
        """Generate a configurations from the Search Tree for Random Search.

        Parameters
        ----------------
        search_tree: SearchTreeType
            Search Tree that defines the search space.
        n: int
            Number of random configurations to generate.
        seed: int
            Seed for the random number generator.

        Returns
        ----------------
        list[BaseConfig]:
            List of configurations generated from the Search Tree in Random Search.
        """
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(),
                search_tree=search_tree,
                random_n=n,
                seed=seed,
            )
        ]

    def get_configs_grid_from_path(self, search_tree_path: Path | str):
        """Generate a grid of configurations from the Search Tree in Grid Search."""
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(),
                search_tree=self._yaml_load(search_tree_path),
            )
        ]

    def yield_configs_grid_from_path(self, search_tree_path: Path | str):
        """Generate a grid of configurations from the Search Tree in Grid Search."""
        for cfg in _yield_config_search_space(
            config_tree=self.to_dict(), search_tree=self._yaml_load(search_tree_path)
        ):
            yield self.from_dict(cfg)

    def get_configs_random_from_path(
        self, search_tree_path: Path | str, n: int, seed: int
    ):
        """Generate random config from the Search Tree in Random Search."""
        return [
            self.from_dict(cfg)
            for cfg in _yield_config_search_space(
                config_tree=self.to_dict(),
                search_tree=self._yaml_load(search_tree_path),
                random_n=n,
                seed=seed,
            )
        ]


def _to_search_tree(obj: BaseConfig) -> SearchTreeType:
    """Convert the configuration class to a search tree."""
    return {
        field.name: (
            _to_search_tree(getattr(obj, field.name))
            if inspect.isclass(field.type) and issubclass(field.type, BaseConfig)
            else (  # Leaves are either [type_hint] or [{possbile enuum values}]
                [str(field.type)]
                if not isinstance(field.type, enum.EnumMeta)
                else [str(set(field.type._value2member_map_.keys())).replace("'", '')]
            )
        )
        for field in dataclasses.fields(obj)
        # We only need to check the fields that are searchable or of type BaseConfig
        # for recursion.
        if (
            (inspect.isclass(field.type) and issubclass(field.type, BaseConfig))
            or field.metadata.get('searchable')
        )
    }


def _yield_config_search_space(
    config_tree: JsonSerializableDict,
    search_tree: SearchTreeType,
    random_n: int | None = None,
    seed: int | None = None,
):
    """
    Generate a grid of configs from the Search Tree in Grid Search or Random Search.

    Parameters
    ----------------
    config_tree: JsonSerializableDict
        Configuration tree to be used as base for the search space.
    search_tree: SearchTreeType
        Search Tree that defines the search space.
    random_n: int | None
        Number of random configurations to generate from the search space.
    seed: int | None
        Seed for the random number generator for reproducibility.

    Notes
    ----------------
    - If `random_n` is given `seed` must also be provided.
    - If they are both `None` the function will generate a grid search space.
    """
    mapping = _get_grid_mapping(search_tree)
    product = itertools.product(*mapping.values())
    if random_n:  # Random Search might get slow for large search spaces.
        product = list(product)
        random.Random(seed).shuffle(product)
        product = product[:random_n]
    for values in product:
        for k, values in zip(mapping.keys(), values):
            set_value_by_path(config_tree, k, values)
        yield config_tree


def _get_grid_mapping(search_tree: SearchTreeType):
    """Get the mapping of the search tree for grid search.

    Returns
    ----------------
    dict:
        Mapping where the keys are point separeted string paths
        and the values are the leafes.
    """
    search_tree = prune_null_path(search_tree)  # Prune the null paths from the tree.
    tree_paths = get_all_string_path(search_tree)  # Get all the paths of the tree.
    mapping = {k: (get_leaf_by_path(search_tree, k)) for k in tree_paths}
    return mapping


def _check_extra_names(config: dict[str, typing.Any], field_names: set[str]):
    """Check for extra field names in the configuration.

    Raises
    ----------------
    ValueError:
        If extra field names are found in the configuration.
    """
    extra_names = set(config.keys()) - field_names
    if extra_names:
        raise ValueError(
            f'Extra fields in the configuration: {extra_names}. '
            f'Allowed fields are: {field_names}'
        )


def prune_null_path(tree: SearchTreeType) -> SearchTreeType:
    """Prune the null values from the search tree.

    Parameters
    ----------------
    tree: SearchTreeType
        Search Tree to prune paths to null leafes.

    Returns
    ----------------
    SearchTreeType:
        Pruned Search Tree with paths to null leafes removed.

    Notes
    ------
    - The Function will prune paths leading to terminataed leafes.
        (from leafes to the root)

    """
    keys_to_delete = []

    for key, branch in tree.items():
        if isinstance(branch, dict):
            prune_null_path(branch)
        if not branch:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del tree[key]
    return tree


def get_leaf_by_path(
    search_tree: SearchTreeType, path: list[str] | str
) -> SearchLeafType:
    """Access a value in the search tree by path."""
    if isinstance(path, str):
        path = path.split('.')
    return get_by_path(search_tree, path)


def set_value_by_path(
    config_dict: JsonSerializableDict, path: list[str] | str, value: JsonType
) -> JsonSerializableDict:
    """Set the Configuration value by path in the configuration dictionary."""
    if isinstance(path, str):
        path = path.split('.')
    return set_by_path(config_dict, path, value)


def get_all_string_path(tree: dict, sep: str = '.') -> list[str]:
    """Build the string path of the tree leafes recursively.

    Parameters
    ----------------
    tree: dict
        Tree to get the string path from.
    sep: str
        Separator for the keys in the path.

    Returns
    ----------------
    list[str]:
        List of string paths of the tree.
    """
    keys = []
    for k, v in tree.items():
        if isinstance(v, dict):
            keys.extend([f'{k}{sep}{kk}' for kk in get_all_string_path(v)])
        else:
            keys.append(k)
    return keys


def get_by_path(tree: dict, path: list):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, path, tree)


def set_by_path(tree: dict, path: list, value):
    """Set a value in a pytree by path."""
    reduce(operator.getitem, path[:-1], tree)[path[-1]] = value
    return tree


Annotation: typing.TypeAlias = (
    type | UnionType | GenericAlias | typing._UnionGenericAlias | typing._GenericAlias
)


def check_type(attr, annot: Annotation) -> bool:
    """Check the type of the attribute recursively.

    Parameters
    ----------------
    attr: Any
        Attribute to be checked.
    annot: Annotation
        Type hint to be checked against the attribute.

    Returns
    ----------------
    bool:
        True if the attribute matches the type hint, False otherwise.

    Examples
    ----------------
    >>> check_type(1, int)
    True
    >>> check_type("abc", int)
    False
    >>> check_type([1, "a"], list[int | str])
    True
    >>> check_type([1, 2, '3'], list[int])
    False
    >>> check_type({'a': [1, 2], 'b': ["a", "b"]}, dict[str, list[int]])
    False
    >>> check_type({'a': [1, 2], 'b': ["a", "b"]}, dict[str, list[int | str]])
    True
    """
    if annot == typing.Any:
        return True
    if isinstance(annot, (UnionType, typing._UnionGenericAlias)):
        return any([check_type(attr, t) for t in typing.get_args(annot)])

    elif isinstance(annot, (GenericAlias, typing._GenericAlias)):
        if issubclass(typing.get_origin(annot), typing.List):
            if not isinstance(attr, typing.List):
                return False
            return all(
                [check_type(element, typing.get_args(annot)[0]) for element in attr]
            )
        elif issubclass(typing.get_origin(annot), typing.Dict):
            if not isinstance(attr, typing.Dict):
                return False
            return all(
                [
                    check_type(key, typing.get_args(annot)[0])
                    and check_type(value, typing.get_args(annot)[1])
                    for key, value in attr.items()
                ]
            )
        else:
            raise ValueError(
                f'Unsupported Generic Type: {typing.get_origin(annot)}. '
                'Contact the maintainer to add more type support.'
            )
    elif isinstance(annot, typing.ForwardRef):
        return check_type(attr, eval(annot.__forward_arg__))

    return isinstance(attr, annot)
