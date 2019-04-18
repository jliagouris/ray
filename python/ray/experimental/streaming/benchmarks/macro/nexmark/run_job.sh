#!/bin/bash

QUERY=$1
THROUGHPUT=$2

if [ $# != 2 ];
then
    echo "Usage: ./run_job.sh <query number> <records>"
    exit
fi

DURATION=60
NUM_EVENTS=$(( $THROUGHPUT * $DURATION ))
FILENAME="latency-"$QUERY"-query-"$THROUGHPUT"-throughput-1-workers.txt"
if ls $FILENAME 1> /dev/null 2>&1
then
    echo "Output $FILENAME already found, skipping..."
    sleep 1
    exit
fi
echo "Logging to $FILENAME"


rm -r /tmp/ray/*
python query_$QUERY.py \
    --bids-file /home/ubuntu/ray/python/ray/experimental/streaming/benchmarks/macro/nexmark/old-bids \
    --auctions-file /home/ubuntu/ray/python/ray/experimental/streaming/benchmarks/macro/nexmark/auctions \
    --persons-file /home/ubuntu/ray/python/ray/experimental/streaming/benchmarks/macro/nexmark/persons \
    --records $NUM_EVENTS --dump-file dump-$QUERY --simulate-cluster \
    --latency-file $FILENAME --source-rate $THROUGHPUT

echo "target throughput: $THROUGHPUT" | tee -a $FILENAME
echo "Query $QUERY `grep "SOURCE THROUGHPUT" /tmp/ray/*/logs/worker*`" | tee -a $FILENAME
