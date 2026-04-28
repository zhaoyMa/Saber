#!/bin/bash

CUDA_VISIBLE_DEVICES="0"      
CONFIG="./configs/humaneval.yaml" 
DEBUG="false"                    

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="eval_${TIMESTAMP}.log"

CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nohup python eval.py --config ${CONFIG}"

CMD="${CMD} > ${LOG_FILE} 2>&1 &"

echo "Starting evaluation..."
echo "Config file: ${CONFIG}"
echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${LOG_FILE}"
echo "Command: ${CMD}"

# Execute command
eval $CMD

# Get process ID
PID=$!
echo "Process ID: ${PID}"
echo "Evaluation started, logs will be written to ${LOG_FILE}"
echo "Use 'tail -f ${LOG_FILE}' to view real-time logs"
echo "Use 'kill ${PID}' to terminate the process"
