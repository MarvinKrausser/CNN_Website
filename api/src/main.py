"""One entrypoint for everything in api/. Run it from the api/ folder:

    python -m src.main                              list all tasks
    python -m src.main yolo.train                   run a task with its defaults
    python -m src.main yolo.train --epochs 20 --batch-size 32
    python -m src.main yolo.train --help            show every option of a task

Defaults live in src/<group>/config.py; edit them there or override them per
run with flags. To start from an IDE "Run" button without arguments, set
DEFAULT_TASK (and optionally DEFAULT_ARGS) below.
"""

import argparse
from dataclasses import fields
import importlib
import sys

# Used when the script is started without arguments.
DEFAULT_TASK = None            # e.g. "yolo.train"
DEFAULT_ARGS = []              # e.g. ["--epochs", "20"]

# group name -> package that contains config.py (Config) and tasks.py (TASKS).
# Packages are only imported when one of their tasks is used, so a missing
# training dependency does not break the others.
GROUPS = {
    "bird": "src.bird_cnn",
    "yolo": "src.yolo",
    "rcnn": "src.r_cnn",
    "server": "src.production",
}


def load_group(group: str):
    package = GROUPS[group]
    config_cls = importlib.import_module(f"{package}.config").Config
    tasks = importlib.import_module(f"{package}.tasks").TASKS
    return config_cls, tasks


def first_line(text):
    return (text or "").strip().splitlines()[0] if text and text.strip() else ""


def list_tasks():
    print(__doc__)
    print("Tasks:")
    for group in GROUPS:
        try:
            _, tasks = load_group(group)
        except ImportError as e:
            print(f"  {group}.*  (unavailable: {e})")
            continue
        for name, fn in tasks.items():
            print(f"  {group + '.' + name:<28} {first_line(fn.__doc__)}")


def build_parser(task: str, config_cls, fn):
    parser = argparse.ArgumentParser(
        prog=f"python -m src.main {task}",
        description=(fn.__doc__ or "").strip(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for f in fields(config_cls):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, dest=f.name, type=f.type, default=f.default, metavar=f.type.__name__.upper())
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv and DEFAULT_TASK:
        argv = [DEFAULT_TASK, *DEFAULT_ARGS]
    if not argv or argv[0] in ("-h", "--help", "list"):
        list_tasks()
        return

    task, *rest = argv
    group, _, name = task.partition(".")
    if group not in GROUPS:
        sys.exit(f"Unknown group '{group}'. Groups: {', '.join(GROUPS)}")

    config_cls, tasks = load_group(group)
    if name not in tasks:
        sys.exit(f"Unknown task '{task}'. Tasks in {group}: {', '.join(tasks)}")

    fn = tasks[name]
    args = build_parser(task, config_cls, fn).parse_args(rest)
    cfg = config_cls(**vars(args))
    fn(cfg)


if __name__ == "__main__":
    main()
