from collections.abc import Hashable
from typing import Any, Generic, TypeVar

K = TypeVar("K", bound=Hashable)  # type of keys, must be hashable, defined by subclass
V = TypeVar("V")  # type of values, defined by subclass


class RegistryMeta(type):
    """Metaclass that initializes a '_registry' attribute for each class.

    This metaclass ensures that any class using it will have a '_registry' attribute,
    which is a dictionary initialized to an empty dictionary. This is useful for creating
    classes that need a registry for storing and retrieving objects by a key.

    Using this metaclass ensures that:
    - Each class, including subclasses, has its own separate '_registry' dictionary.
    - Registrations in one subclass do not affect other subclasses, avoiding potential conflicts.

    Attributes:
        _registry: A dictionary to store registered objects.

    Methods:
        __new__(cls, name, bases, dct): Modifies the class creation process to include a new '_registry' attribute.
    """

    def __new__(cls: type["RegistryMeta"], name: str, bases: tuple[Any], dct: dict[Any, Any]) -> "RegistryMeta":
        """Create a new class with a '_registry' attribute.

        Args:
            name: The name of the class.
            bases: The base classes of the class.
            dct: The class attributes.

        Returns:
            A new class with a '_registry' class attribute.
        """
        dct["_registry"] = dict()
        return super().__new__(cls, name, bases, dct)


class Registry(Generic[K, V], metaclass=RegistryMeta):
    """A registry for storing and retrieving objects by key."""

    _registry: dict[K, V] = dict()

    @classmethod
    def register(cls, key: K, value: V, overwrite: bool = False) -> None:
        """Register a value with the registry.

        Args:
            key: The key to register the value under.
            value: The value to register.
            overwrite: If True, overwrite the value if it already exists. If False, raise an error if the
                value already exists.

        Raises:
            ValueError: If the value already exists and overwrite is False.
        """

        if key in cls._registry and not overwrite:
            raise ValueError(f"Key already registered: {key}")
        cls._registry[key] = value

    @classmethod
    def get(cls, key: K) -> V:
        """Get a value from the registry.

        Args:
            key: The key to get the value for.

        Returns:
            The value for the specified key.

        Raises:
            ValueError: If no value is registered for the key.
        """

        if key not in cls._registry:
            raise ValueError(f"No value registered for key: {key}")
        return cls._registry[key]  # type: ignore

    @classmethod
    def get_registered_keys(cls) -> list[K]:
        """Get a list of all registered keys.

        Returns:
            A list of all registered keys.
        """

        return list(cls._registry.keys())
