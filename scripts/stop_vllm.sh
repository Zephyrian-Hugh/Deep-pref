#!/bin/bash

# Script to stop vLLM servers - Ultimate Version (based on process groups)
#
# Core principle:
# 1. Find the main process PID by listening port number.
# 2. Find the process group ID (PGID) from the main process PID.
# 3. Use `kill -- -<PGID>` to send signals to the entire process group,
#    which will kill all processes (parent, child, grandchild...) in the group.

echo "🚀 Preparing to stop vLLM servers..."

# Function to stop process group by port
stop_process_group_by_port() {
    local port=$1
    local name=$2
    
    echo ""
    echo "--- Processing port $port ($name) ---"
    
    # Use lsof to find process PID listening on specified port
    # -t: only output PID
    # -i: specify network connection
    local pid=$(lsof -t -i:$port)

    if [ -z "$pid" ]; then
        echo "✅ No process found listening on port $port."
        return
    fi

    # Handle potential multiple PIDs (though rare)
    pid=$(echo "$pid" | head -n 1)
    echo "🔍 Found main process PID: $pid listening on port $port"

    # Get process group ID (PGID) through PID
    # ps -o pgid= -p <PID> gets PGID precisely
    local pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')

    if [ -z "$pgid" ]; then
        echo "⚠️ Could not get PGID for PID $pid. Will try to kill the process directly."
        kill -9 "$pid"
        return
    fi

    echo "🎯 Target process group ID (PGID): $pgid"
    echo "💥 Preparing to terminate entire process group..."

    # Step 1: Send SIGTERM (signal 15) to try graceful termination of process group
    # Note `kill` command's `--` and PGID's `-` - this indicates operation on process group
    echo "  1. Sending SIGTERM signal, requesting graceful exit..."
    kill -TERM -- "-$pgid"
    
    # Wait a few seconds for graceful exit
    sleep 3

    # Step 2: Check if process group still exists, if yes, force kill
    # `kill -0` is used to check process existence
    if kill -0 -- "-$pgid" 2>/dev/null; then
        echo "  2. Process group still exists, sending SIGKILL signal for forced termination..."
        kill -KILL -- "-$pgid"
        sleep 2
        echo "  Forced termination complete."
    else
        echo "  Process group exited successfully."
    fi

    echo "✅ $name server (port $port) related processes have been cleaned up."
}

# --- Main Execution Flow ---

# Stop Policy Model server
stop_process_group_by_port 8000 "Policy Model"

# Stop Reward Model server
stop_process_group_by_port 8001 "Reward Model"

echo ""
echo "--- Final Check ---"

# Check GPU usage
echo "📊 Current GPU process status:"
# Use `|| true` to ensure script doesn't exit on error when no GPU processes exist
gpu_procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits)
if [ -z "$gpu_procs" ]; then
    echo "  No compute processes running on GPU."
else
    echo "$gpu_procs"
fi

# Check port usage
echo ""
echo "🔌 Current port listening status:"
if lsof -i :8000 -i :8001 >/dev/null 2>&1; then
    echo "  ⚠️ Warning: Port 8000 or 8001 is still in use!"
    lsof -i :8000 -i :8001
else
    echo "  Ports 8000 and 8001 are released."
fi

# Clean up log files option
echo ""
read -p "Delete log files (policy_vllm.log, reward_vllm.log)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f policy_vllm.log reward_vllm.log
    echo "🗑️ Log files deleted."
fi

echo ""
echo "🎉 All operations completed!"