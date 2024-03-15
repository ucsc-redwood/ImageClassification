#!/bin/bash
DIRECTORY="/home/riksharm/ImageClassification/extract"

CORES="0-7"  # Adjust this to set the desired cores

count=0  # Initialize iteration count
total_time_ms=0  # Initialize total time counter

for FILE in $DIRECTORY/*.txt; do
    count=$((count + 1))  # Increment the iteration count

    echo "Processing file: $FILE"
    echo "File count: $count"

    # Execute the C++ program and capture its output
    output=$(taskset -c $CORES ./a.out "$FILE")

    # Display the output
    echo "$output"

    # Extract the total time from the output and remove the 'ms' suffix
    cpp_time_ms=$(echo "$output" | grep "Total time:" | awk '{print $3}' | sed 's/ms//')

    # Add the time to the total time counter
    total_time_ms=$(echo "$total_time_ms + $cpp_time_ms" | bc)

    # Output the cumulative time taken so far
    echo "Total time taken up to now: $total_time_ms milliseconds"
    echo "-------------------------------------"
done

# Output the total time taken for all C++ executions at the end
echo "Total C++ execution time: $total_time_ms milliseconds"

