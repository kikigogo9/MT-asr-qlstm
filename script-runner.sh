#!/bin/bash
source venv3/bin/activate

while true; do
    python3 gridsearch.py "8" "0.001" "reupload" "4" "yes"
    echo "Restarting..."
    sleep 1
done