from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


class AddTuringOperationsToSearchPath(SearchPathPlugin):

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="example-searchpath-plugin", path="pkg://tiny_moves/operations")


def register_plugins() -> None:
    """Hydra users should call this function before invoking @hydra.main."""
    Plugins.instance().register(AddTuringOperationsToSearchPath)
