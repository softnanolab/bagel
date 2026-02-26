"""Command line interface for BAGEL."""

from __future__ import annotations

import argparse
import json
import sys

from .config import (
    ConfigCompilationError,
    ConfigLoadError,
    compile_loaded_config,
    load_config,
    run_compiled,
    schema_json,
    validate_loaded_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='bagel', description='BAGEL command line interface')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Run BAGEL from a YAML config file')
    run_parser.add_argument('config', type=str, help='Path to YAML config file')

    validate_parser = subparsers.add_parser('validate', help='Validate BAGEL YAML config without running optimization')
    validate_parser.add_argument('config', type=str, help='Path to YAML config file')

    schema_parser = subparsers.add_parser('schema', help='Export JSON schema for BAGEL config format')
    schema_parser.add_argument('--version', type=int, default=1, help='Schema version to export (default: 1)')

    return parser


def _cmd_run(config_path: str) -> int:
    loaded = load_config(config_path)
    compiled = compile_loaded_config(loaded)
    _ = run_compiled(compiled)
    print(str(compiled.minimizer.log_path))
    return 0


def _cmd_validate(config_path: str) -> int:
    loaded = load_config(config_path)
    validate_loaded_config(loaded)
    print('OK')
    return 0


def _cmd_schema(version: int) -> int:
    schema = schema_json(version)
    print(json.dumps(schema, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == 'run':
            return _cmd_run(args.config)
        if args.command == 'validate':
            return _cmd_validate(args.config)
        if args.command == 'schema':
            return _cmd_schema(args.version)
    except (ConfigLoadError, ConfigCompilationError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
