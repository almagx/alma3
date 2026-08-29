#!/bin/sh
set -eu

case "${1:-}" in
    "") set -- alma3 --help ;;
    demo|download|infer|verify-release|-h|--help|--version) set -- alma3 "$@" ;;
esac

exec "$@"
