#!/bin/bash
DIRECTORY="/home/riksharm/ImageClassification/extract"

for FILE in $DIRECTORY/*.txt; do
    echo "Processing file: $FILE"
    ./a.out "$FILE"
    echo "-------------------------------------"
done

