#!/bin/bash

# Script to start vLLM servers
# This script will start two vLLM servers in the background:
# - Policy model on GPU 0, port 8000
# - Reward model on GPU 1, port 8001

echo "🚀 Preparing to start vLLM servers..."
echo "Note: This script will automatically start two vLLM servers in the background"
echo "Policy model on GPU 0, port 8000"
echo "Reward model on GPU 1, port 8001"
echo "Each model uses a single GPU to avoid memory competition"
echo ""

# Check if ports are already in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        echo "✅ Port $1 is available"
        return 0
    fi
}

echo "Checking port availability..."
check_port 8000
POLICY_PORT_OK=$?
check_port 8001 
REWARD_PORT_OK=$?

if [ $POLICY_PORT_OK -ne 0 ] || [ $REWARD_PORT_OK -ne 0 ]; then
    echo "❌ Please close any processes using ports 8000 or 8001"
    exit 1
fi

echo ""
echo "🔄 Starting Policy Model vLLM Server (GPU 0, port 8000)..."

# Start Policy Model vLLM Server
CUDA_VISIBLE_DEVICES=0 \
NCCL_DEBUG=INFO \
NCCL_CUMEM_ENABLE=0 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NET=Socket \
NCCL_SOCKET_IFNAME=lo \
GLOO_SOCKET_IFNAME=lo \
no_proxy=localhost,127.0.0.1 \
nohup trl vllm-serve \
  --model ./models/policy_model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --port 8000 \
  --trust-remote-code \
  > policy_vllm.log 2>&1 &

POLICY_PID=$!
echo "Policy Model vLLM Server starting... PID: $POLICY_PID"

# Wait for Policy server to start
sleep 10

echo "🔄 Starting Reward Model vLLM Server (GPU 1, port 8001)..."

# Start Reward Model vLLM Server - use different NCCL settings to avoid conflicts
# Use vllm.entrypoints.openai.api_server to provide OpenAI-compatible API
CUDA_VISIBLE_DEVICES=1 \
NCCL_DEBUG=INFO \
NCCL_CUMEM_ENABLE=0 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_NET=Socket \
NCCL_SOCKET_IFNAME=lo \
GLOO_SOCKET_IFNAME=lo \
no_proxy=localhost,127.0.0.1 \
nohup python -m vllm.entrypoints.openai.api_server \
  --model ./models/reward_model \
  --served-model-name reward_model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --host 127.0.0.1 \
  --port 8001 \
  --trust-remote-code \
  > reward_vllm.log 2>&1 &

REWARD_PID=$!
echo "Reward Model vLLM Server starting... PID: $REWARD_PID"

echo ""
echo "📝 Server Information:"
echo "Policy Model: PID=$POLICY_PID, log file=policy_vllm.log"
echo "Reward Model: PID=$REWARD_PID, log file=reward_vllm.log"
echo ""
echo "⏳ Waiting for servers to start (about 2-3 minutes)..."

# Wait for servers to start
wait_for_server() {
    local port=$1
    local name=$2
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "✅ $name server started successfully (port $port)"
            return 0
        fi
        sleep 5
        attempt=$((attempt + 1))
        echo "   Waiting for $name server to start... ($attempt/$max_attempts)"
    done
    
    echo "❌ $name server startup timed out"
    return 1
}

# Wait for both servers to start
wait_for_server 8000 "Policy Model"
POLICY_STATUS=$?

wait_for_server 8001 "Reward Model" 
REWARD_STATUS=$?

echo ""
if [ $POLICY_STATUS -eq 0 ] && [ $REWARD_STATUS -eq 0 ]; then
    echo "🎉 All vLLM servers started successfully!"
    
    # Test reward model server
    echo "🧪 Testing Reward Model server..."
    if python scripts/test_reward_vllm.py http://127.0.0.1:8001; then
        echo "✅ Reward Model server test passed!"
    else
        echo "⚠️  Reward Model server test failed, but servers are running"
    fi
    
    echo ""
    echo "You can now run the training script:"
    echo "bash scripts/run_training.sh"
    echo ""
    echo "To stop the servers, run:"
    echo "bash scripts/stop_vllm.sh"
else
    echo "❌ Server startup failed, please check log files"
    echo "Policy Model log: cat policy_vllm.log"
    echo "Reward Model log: cat reward_vllm.log"
fi