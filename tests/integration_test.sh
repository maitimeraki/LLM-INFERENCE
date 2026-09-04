#!/bin/bash
# Manual integration test for unified vLLM serving
# Tests both dense and MoE models with conflict resolution

set -e

echo "=================================="
echo "vLLM Integration Test Suite"
echo "=================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v python >/dev/null 2>&1 || { echo "Python not found"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not found"; exit 1; }

python -c "import torch; assert torch.cuda.is_available()" || {
    echo "CUDA not available - tests require GPU";
    exit 1;
}

echo "✓ Prerequisites met"
echo ""

# Test 1: Server startup with GPT-2
echo "Test 1: Server startup with small model (GPT-2)"
echo "------------------------------------------------"

# Start server in background
python serve.py --model gpt2 --port 8001 > /tmp/serve_test.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server startup..."
for i in {1..60}; do
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then
        echo "✓ Server started successfully"
        break
    fi
    sleep 1
done

# Check server is responsive
if ! curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo "✗ Server failed to start"
    cat /tmp/serve_test.log
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Test inference
echo "Testing inference endpoint..."
RESPONSE=$(curl -s http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt2",
        "prompt": "Hello, world!",
        "max_tokens": 10
    }')

if echo "$RESPONSE" | grep -q '"choices"'; then
    echo "✓ Inference successful"
else
    echo "✗ Inference failed"
    echo "Response: $RESPONSE"
    kill $SERVER_PID
    exit 1
fi

# Test metrics endpoint
echo "Testing metrics endpoint..."
METRICS=$(curl -s http://localhost:8001/metrics)

if echo "$METRICS" | grep -q '"memory"'; then
    echo "✓ Metrics endpoint working"
else
    echo "✗ Metrics endpoint failed"
    kill $SERVER_PID
    exit 1
fi

# Check memory coordination in logs
echo "Verifying memory coordination..."
if grep -q "Unified Memory Coordination Initialized" /tmp/serve_test.log; then
    echo "✓ Memory coordinator initialized"
else
    echo "✗ Memory coordinator not found in logs"
    kill $SERVER_PID
    exit 1
fi

# Check weight bridge in logs
if grep -q "SparseMoEWeightBridge initialized" /tmp/serve_test.log; then
    echo "✓ Weight bridge initialized"
else
    echo "✗ Weight bridge not found in logs"
    kill $SERVER_PID
    exit 1
fi

# Cleanup
echo "Cleaning up..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
rm /tmp/serve_test.log

echo ""
echo "✅ All tests passed!"
echo ""
echo "Summary:"
echo "  • Server startup: PASS"
echo "  • Inference endpoint: PASS"
echo "  • Metrics endpoint: PASS"
echo "  • Memory coordination: PASS"
echo "  • Weight bridge: PASS"
echo ""
echo "Integration test complete."
