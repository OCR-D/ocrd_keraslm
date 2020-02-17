#!/bin/bash

set -e -x

CACHE_DIR=/tmp/ocrd_keraslm-cache
TMP_DIR=$(mktemp -d -t ocrd_keraslm-tmp-XXXXXXXXXX)
GT_FILES="kant_aufklaerung_1784 loeber_heuschrecken_1693"

trap "rm -fr '$TMP_DIR'" ERR

test -d "$1" # target directory must already exist

test -d "$CACHE_DIR" || mkdir -p "$CACHE_DIR"

for GT_FILE in $GT_FILES; do
    test -f "$CACHE_DIR/$GT_FILE" ||
        wget -P "$CACHE_DIR" http://www.deutsches-textarchiv.de/book/download_txt/$GT_FILE
    sed -e '//d;/^[[].*[]]$/d' < "$CACHE_DIR/$GT_FILE" > "$TMP_DIR/${GT_FILE}.txt"
done

mv "$TMP_DIR"/*.txt "$1"




