#!/bin/sh
set -eu

case "${1:-}" in
    "") set -- alma3 --help ;;
    demo|infer|verify-release|export-bedmethyl-target|-h|--help|--version) set -- alma3 "$@" ;;
esac

exec "$@"
