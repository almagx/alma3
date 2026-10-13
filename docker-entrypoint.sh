#!/bin/sh
set -eu

case "${1:-}" in
    "") set -- alma3 --help ;;
    demo|infer|verify-release|export-bedmethyl-target|-h|--help|--version) set -- alma3 "$@" ;;
esac

if [ "$(id -u)" = 0 ]; then
    owner=$(stat -c '%u:%g' .)
    set -- -- "$@"
    if [ "$owner" != "0:$(id -g)" ]; then
        set -- --reuid="${owner%:*}" --regid="${owner#*:}" --clear-groups "$@"
    fi
    exec setpriv --bounding-set=-dac_override,-dac_read_search "$@"
fi

exec "$@"
