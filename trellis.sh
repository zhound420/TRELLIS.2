#!/bin/bash

# TRELLIS.2 Control Script
# Interactive menu to start/stop/check status of the Gradio app

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="/tmp/trellis_app.pid"
LOG_FILE="/tmp/trellis_app.log"
PORT=7860

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

is_port_listening() {
    ss -tlnp 2>/dev/null | grep -q ":$PORT " && return 0
    return 1
}

start_app() {
    if is_running; then
        echo -e "${YELLOW}App is already running (PID: $(cat $PID_FILE))${NC}"
        return 1
    fi

    echo "Starting TRELLIS.2 app..."

    # Source conda and activate environment
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

    conda activate trellis2

    # Set environment variables
    export OPENCV_IO_ENABLE_OPENEXR=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # Start the app
    cd "$SCRIPT_DIR"
    nohup python app.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo "Started with PID $(cat $PID_FILE)"
    echo "Waiting for app to be ready..."

    # Wait for port to be ready (max 3 minutes)
    for i in {1..180}; do
        if is_port_listening; then
            echo -e "${GREEN}App is ready at http://127.0.0.1:$PORT${NC}"
            return 0
        fi
        if ! is_running; then
            echo -e "${RED}App failed to start. Check logs: $LOG_FILE${NC}"
            rm -f "$PID_FILE"
            return 1
        fi
        sleep 1
        printf "\rWaiting... %ds" "$i"
    done

    echo -e "\n${YELLOW}App is still starting. Check status later.${NC}"
}

stop_app() {
    if ! is_running; then
        echo -e "${YELLOW}App is not running${NC}"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid=$(cat "$PID_FILE")
    echo "Stopping app (PID: $pid)..."

    # Try graceful shutdown first
    kill "$pid" 2>/dev/null
    sleep 3

    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Force killing..."
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi

    rm -f "$PID_FILE"
    echo -e "${GREEN}App stopped${NC}"
}

show_status() {
    echo ""
    echo "=== TRELLIS.2 Status ==="

    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo -e "Process: ${GREEN}Running${NC} (PID: $pid)"

        # Show memory usage
        local mem=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
        echo "Memory:  $mem"
    else
        echo -e "Process: ${RED}Not running${NC}"
    fi

    if is_port_listening; then
        echo -e "Port:    ${GREEN}Listening on $PORT${NC}"
        echo -e "URL:     http://127.0.0.1:$PORT"
    else
        echo -e "Port:    ${RED}Not listening${NC}"
    fi

    echo "Log:     $LOG_FILE"
    echo ""
}

show_menu() {
    echo ""
    echo "=========================="
    echo " TRELLIS.2 Control Panel"
    echo "=========================="
    echo "1) Start app"
    echo "2) Stop app"
    echo "3) Check status"
    echo "4) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Select option [1-4]: " choice
    echo ""

    case $choice in
        1) start_app ;;
        2) stop_app ;;
        3) show_status ;;
        4) echo "Goodbye!"; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
done
