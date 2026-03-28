from __future__ import annotations

import sys

from setuptools import setup

from _native_build import get_setup_kwargs


def main(argv: list[str] | None = None) -> None:
    script_args = list(argv or sys.argv[1:])
    if not script_args:
        script_args = ["build_ext", "--inplace"]
    setup(
        name="nepactive-native-nep",
        packages=["nepactive"],
        package_dir={"nepactive": "."},
        script_args=script_args,
        **get_setup_kwargs(),
    )


if __name__ == "__main__":
    main()
