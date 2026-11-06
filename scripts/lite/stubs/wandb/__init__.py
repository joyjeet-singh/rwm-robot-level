"""No-op stand-in for the `wandb` package (no network, no files). Only what the lite code touches:
init, log, finish, config, Table, run. `run` is an object (not None) so that `wandb.run is not None`
branches take the same path as with the real package; its `name` can be set."""


class _Config(dict):
    def update(self, *a, **k):          # wandb.config.update(config.to_dict())
        return None


class _Run:
    name = None
    id = "stub"
    dir = None

    def log(self, *a, **k):
        return None

    def finish(self, *a, **k):
        return None


class Table:
    def __init__(self, *a, **k):
        pass

    def add_data(self, *a, **k):
        return None


config = _Config()
run = _Run()


def init(*a, **k):
    return run


def log(*a, **k):
    return None


def finish(*a, **k):
    return None
