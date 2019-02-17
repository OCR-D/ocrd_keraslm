#!/bin/bash

set -e -x

CACHE_DIR=/tmp/ocrd_keraslm-cache
TMP_DIR=$(mktemp -d -t ocrd_keraslm-tmp-XXXXXXXXXX)
GT_FILES="kant_aufklaerung_1784 loeber_heuschrecken_1693"

trap "rm -fr '$TMP_DIR'" ERR

test ! -e "$1" # target directory must already exist

test -d "$CACHE_DIR" || mkdir -p "$CACHE_DIR"

for GT_FILE in $GT_FILES; do
    test -f "$CACHE_DIR/$GT_FILE" ||
        wget -P "$CACHE_DIR" http://www.deutsches-textarchiv.de/book/download_txt/$GT_FILE
    sed -e '//d;/^[[].*[]]$/d' < "$CACHE_DIR/$GT_FILE" > "$TMP_DIR/${GT_FILE}.txt"
done

cat <<EOF > "$TMP_DIR/page-extract-imagefilename.xsl"
<xsl:stylesheet
    version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:pc="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <!-- unfortunately, this totally depends on the exact namespace string, i.e. version -->
  <!-- rid of xml syntax: -->  
  <xsl:output
      method="text"
      standalone="yes"
      omit-xml-declaration="yes"/>
  <!-- get imageFilename attribute verbatim: -->  
  <xsl:template match="pc:PcGts/pc:Page">
    <xsl:value-of select="@imageFilename" disable-output-escaping="yes"/>
    <xsl:apply-templates/>
  </xsl:template>
  <!-- override implicit rules copying elements and attributes: -->
  <xsl:template match="text()"/>
</xsl:stylesheet>
EOF

for GT_FILE in $GT_FILES; do
    test -f "$CACHE_DIR/${GT_FILE}.zip" ||
        wget -P "$CACHE_DIR" http://www.ocr-d.de/sites/all/GTDaten/${GT_FILE}.zip
    unzip -d "$TMP_DIR" "$CACHE_DIR/${GT_FILE}.zip"
    pushd "$TMP_DIR/$GT_FILE/$GT_FILE"
    ocrd workspace init .
    ZEROS=0000
    i=0
    for PAGE_FILE in page/*.xml; do
        i=$((i+1))
        ID=${ZEROS:0:$((4-${#i}))}$i
        IMG_FILE=$(xsltproc "$TMP_DIR/page-extract-imagefilename.xsl" "$PAGE_FILE")
        test -f "$IMG_FILE"
        ocrd workspace add -G OCR-D-IMG -i OCR-D-IMG_$ID -g OCR-D-IMG_$ID -m image/tiff "$IMG_FILE"
        ocrd workspace add -G OCR-D-GT-PAGE -i OCR-D-GT-PAGE_$ID -g OCR-D-IMG_$ID -m application/vnd.prima.page+xml "$PAGE_FILE"
    done
    popd
done

# this would break URIs: (still true for ocrd v0.15.2 !!)
#mv "$TMP_DIR" "$1" # atomic
# clone+cp instead:
trap "rm -fr '$TMP_DIR' '$1'" ERR
mkdir -p "$1"
for GT_FILE in $GT_FILES; do # not so atomic
    WORKSPACE="$TMP_DIR/$GT_FILE/$GT_FILE"
    ocrd workspace clone -l "$WORKSPACE/mets.xml" "$1/$GT_FILE"
    # workaround for OCR-D/core/issues/176 (still true for ocrd v0.15.2 !!)
    for PAGE_FILE in "$1/$GT_FILE/OCR-D-GT-PAGE/"*.xml; do
        sed -ie "s|imageFilename=\"|imageFilename=\"file://$PWD/$1/$GT_FILE/OCR-D-IMG/|" "$PAGE_FILE"
    done
    cp "$TMP_DIR/${GT_FILE}.txt" "$1"
done
rm -fr "$TMP_DIR"



