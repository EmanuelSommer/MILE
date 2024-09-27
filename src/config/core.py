"""Top level configuration class for the mile package."""
import dataclasses
import importlib
import inspect
import json
import logging
import logging.config
import logging.handlers
import sys
import time
from dataclasses import field
from pathlib import Path

import flax.linen as nn

from src.config.base import BaseConfig
from src.config.data import DataConfig
from src.config.models.base import ModelConfig
from src.config.training import TrainingConfig
from src.exceptions import MissingConfigError, ModelNotFoundError

models_module = importlib.import_module('src.models')


@dataclasses.dataclass(frozen=True)
class Config(BaseConfig):
    """Top level configuration class for the mile package.

    Notes
    ------
    - This is the top level configuration class for the mile package.
    - It contains 3 main Branches of configuration:
        - Data: Configuration for the Dataset `./data.py`.
        - Model: Configuration for the ML Model `./models/*.py`.
        - Training: Configuration for Training / Sampling Proccess `./training.py`.

    Example::
    ```
    # Creating a Template for specific model, fill-out the template and load it.
    from src.config.core import Config
    # Look-Up the available models
    Config.list_avaliable_models()
    >>> ['LeNet', 'FCN']
    # Creating a Template
    Config.template_to_yaml('config.yaml', 'LeNet')
    # Fill-out the `config.yaml` file and load it
    config = Config.from_yaml('config.yaml')
    ```
    """

    saving_dir: str = field(
        metadata={'description': 'Directory to save checkpoints and logs.'},
    )
    experiment_name: str = field(metadata={'description': 'Name of the experiment.'})
    data: DataConfig = field(metadata={'description': 'Configuration for the Dataset.'})
    model: ModelConfig = field(metadata={'description': 'Configuration for the Model.'})
    training: TrainingConfig = field(
        default_factory=TrainingConfig,
        metadata={'description': 'Configuration for Training.'},
    )
    rng: int = field(
        default=42, metadata={'description': 'RNG Seed.', 'searchable': True}
    )
    logging: bool = field(
        default=True, metadata={'description': 'Enable logging to file and stdout.'}
    )

    def __post_init__(self):
        """Ensure that the saving directory exists."""
        super().__post_init__()
        self._modify_field(**{'saving_dir': Path(self.saving_dir).as_posix()})

    def __hash__(self):
        """Return the hash of the configuration.

        Notes
        -----
        - This might not be the most stable but it is good enough for
            the purpose of keeping unique configurations, when performing
            Grid or Random Search.
        """
        return hash(self.__str__())

    @classmethod
    def to_template(cls, model: str | nn.Module):
        """Return tempalate config for the chosen model.

        Parameters
        ----------
            model: str | nn.Module
                The model to get the template for, If the model is pre-defined
                in the `__init__.py` file of `src.models` submodule,
                it can be referenced by a string name. Otherwise, pass the model
                class directly to generate the template.

        Notes
        -----
        - Each Machine Learning Model in the `mile` franemwork must have a
            `config` field with corresponding `ModelConfig` subclass as its type,
            which containts the configurations the user wants to configure from the
            configuration file. Next to the `config` field the user can add as many
            extra fields as needed, as far as they have default values.

        - As you can see in the example below, the `LeNet` model has a `config` field
            of type `LeNetConfig` which is a subclass of `ModelConfig` and contains
            a Field `model` which exactly matches the `class.__name__` of the model
            definition `LeNet`. it additionally adds some extra fields we do not
            intend to configure from the config file.

        Example::
        ```python
        # Defining Configuration
        from dataclasses import dataclass, field
        from src.config.models.base import ModelConfig, Activation

        @dataclass(frozen=True)
        class LeNetConfig(ModelConfig):
            model: str = 'LeNet'
            activation: Activation = field(default=Activation.SIGMOID)
            use_bias: bool = True
            outdim: int = 10

        # Defining Model
        import flax.linen as nn
        import jax.numpy as jnp

        class LeNet(nn.Module):
            config: LeNetConfig
            some_fix_param: int = 10
            some_other_fix_param = nn.Dense(10)

            @nn.compact
            def __call__(self, x: jnp.ndarray):
                '''
                Forward pass.

                Parameters
                    x (jnp.ndarray):
                        The input data of shape (batch_size, channels, height, width).
                '''
                act = self.config.activation.flax_activation
                x = x.transpose((0, 2, 3, 1))
                x = nn.Conv(
                    features=6, kernel_size=(5, 5), strides=(1, 1), name='conv1'
                )(x)
                x = nn.avg_pool(
                    act(x), window_shape=(2, 2), strides=(2, 2), padding='VALID'
                )
                x = nn.Conv(
                    features=16, kernel_size=(5, 5), strides=(1, 1), name='conv2'
                )(x)
                x = nn.avg_pool(
                    act(x), window_shape=(2, 2), strides=(2, 2), padding='VALID'
                )
                x = x.reshape((x.shape[0], -1))
                x = nn.Dense(features=120, use_bias=self.config.use_bias, name='fc1')(x)
                x = nn.Dense(
                    features=84, use_bias=self.config.use_bias, name='fc2'
                )(act(x))
                x = nn.Dense(
                    features=self.config.outdim,
                    use_bias=self.config.use_bias,
                    name='fc3'
                )(act(x))
                return x
        """
        if isinstance(model, str):
            if model not in models_module.__all__:
                raise ModelNotFoundError(
                    f'In order to access model with a string name, it must be '
                    f'contained in {models_module.__name__} module. Got {model} but '
                    f'expected one of: {models_module.__all__}'
                    f'For custom models, pass the model class directly.'
                )
            model = getattr(models_module, model)
        cfg_cls = cls._get_model_config_cls(model)
        schema = cls.to_schema()
        schema['model'] = cfg_cls.to_schema()
        return schema

    @classmethod
    def template_to_yaml(cls, path: Path | str, model: str | nn.Module):
        """Write the template to a YAML file.

        Notes
        -----
        - See the classmethod `Config.to_template()` for more information.§
        """
        schema = cls.to_template(model)
        cls._yaml_dump(schema, path=path)

    @classmethod
    def template_to_json(cls, path: Path | str, model: str | nn.Module):
        """Write the templat to JSON.

        Notes
        -----
        - See the classmethod `Config.to_template()` for more information.§
        """
        schema = cls.to_template(model)
        json.dump(schema, open(path, 'w'), indent=3)

    @staticmethod
    def list_avaliable_models():
        """List all available pre-defined models.

        Notes
        -----
        - Models must be defined in the __init__.py file of
            `src.models` submodule.
        """
        return models_module.__all__

    def get_flax_model(self, **kwargs) -> nn.Module:
        """
        Return the associated Flax model instance.

        Parameters
        ----------
            **kawrgs:
                Additional keyword Fields other than config
                to pass to the nn.Module constructor.

        Notes
        -----
        - Models must be defined in the __init__.py file of
            `src.models` submodule.
        """
        return getattr(models_module, self.model.model)(config=self.model, **kwargs)

    def setup_dir(self):
        """Setups the saving directory and logging for the experiment.

        Notes
        -----
        - If the experiment_name already exists in given saving directory,
            it will append the current timestamp to the name to make it unique
            and avoid overwriting results.
        - Method Recursively creates the saving directory if it does not exist.
        - If logging is enabled, it will setup logging to file and stdout.
        """
        if self.experiment_dir.exists():
            name = f"{self.experiment_name}_{time.strftime('%Y%m%d-%H%M%S')}"
            self._modify_field(**{'experiment_name': name})
        self.experiment_dir.mkdir(parents=True)
        self.to_yaml(self.experiment_dir / 'config.yaml')
        self._setup_logging()

    @property
    def jax_rng(self):
        """Return the JAX RNG key for the given seed value."""
        try:
            import jax  # type: ignore
        except ImportError:
            raise ImportError('Install "jax" library to use JAX RNG.')
        return jax.random.PRNGKey(self.rng)

    @property
    def n_chains(self):
        """Return the number of chains."""
        return self.training.sampler.n_chains

    @property
    def experiment_dir(self):
        """Return the saving path."""
        return Path(self.saving_dir) / self.experiment_name

    def _setup_logging(self):
        """Setups logging to file and stdout."""
        handlers = []
        handlers.append(logging.FileHandler(self.experiment_dir / 'training.log'))
        if self.logging:
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(
            handlers=handlers,
            level=logging.INFO,
            force=True,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logging.getLogger(__name__).info('Logging successfully setup.')

    @classmethod
    def _handle_field(cls, f: dataclasses.Field, value):
        """Extend the handle_field method to handle ModelConfig fields.

        Notes
        -----
        - We extend the default `handle_field()` method to handle `ModelConfig` fields.
        - When we want to parse the `ModelConfig` field, we need to somehow
            connect the string name to the actual model configuration class.
        - For this Purpose each `ModelConfig` has a `model` field which is a string
            exactly matching the class name of the model it intends to configure.
        - We use this string to get the corresponding `ModelConfig` subclass
            and parse the dictionary accordingly.
        """
        if inspect.isclass(f.type) and issubclass(f.type, ModelConfig):
            if isinstance(value, dict):
                mapping = f.type.get_name_mapping()
                model = mapping.get(value.get('model'))
                if not model:
                    raise ModelNotFoundError(
                        f'Could not find model {value.get("model")}. '
                        f'Avaliable models: {[*mapping.keys()]}'
                    )
                return model.from_dict(value)
        return super()._handle_field(f, value)

    @staticmethod
    def _get_model_config_cls(model: nn.Module) -> type[ModelConfig]:
        """Return the corresponding ModelConfig class for the given model.

        Notes
        _____
        - We use this method for templating purposes in `to_template()` method.
        - Each Machine Learning Model in the `mile` franemwork has a
            `config` field with corresponding `ModelConfig` subclass as its type.
            This way we can map `ModelConfig`-s to corresponding `nn.Module`-s.
        """
        if not inspect.isclass(model) and not issubclass(model, nn.Module):
            raise ValueError(f'Expected a flax.nn.Module, got {type(model)} instead.')
        if 'config' not in model.__dataclass_fields__:
            raise MissingConfigError(
                f'The model {model.__name__} does not have a `config` field '
                'of corresponding pre-defined `ModelConfig` subclass.'
            )
        return model.__dataclass_fields__['config'].type
