#!/bin/sh
set -eu

case "${1:-}" in
    "") set -- alma3 --help ;;
    infer|verify-release|-h|--help) set -- alma3 "$@" ;;
esac

exec "$@"
