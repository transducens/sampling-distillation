#!/bin/bash

input_file=$1
output_file=$2
samples=$3

while IFS= read -r line; do
    for ((i=1; i<=$samples; i++)); do
        echo "$line" >> "$output_file"
    done
done < "$input_file"
