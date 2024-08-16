#!/bin/bash

SESSION_NAME="llava"

# Check if the session exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME exists. Killing it..."
    tmux kill-session -t $SESSION_NAME
fi

# Create a new tmux session
tmux new-session -d -s $SESSION_NAME

# Split the window
tmux split-window -v -t $SESSION_NAME
tmux split-window -h -t $SESSION_NAME

# Send commands to each pane
tmux send-keys -t $SESSION_NAME:0.0 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate llava && export PYTHONPATH=/home/$USER/Documents/LLaVA/:$PYTHONPATH && python3 llava/serve/controller.py" C-m

tmux send-keys -t $SESSION_NAME:0.1 "sleep 5 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate llava && export PYTHONPATH=/home/$USER/Documents/LLaVA/:$PYTHONPATH && python3 llava/serve/model_worker.py --model-path liuhaotian/llava-v1.6-34b" C-m

tmux send-keys -t $SESSION_NAME:0.2 "sleep 40 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate llava && export PYTHONPATH=/home/$USER/Documents/LLaVA/:$PYTHONPATH && python3 llava/serve/gradio_web_server.py" C-m

# Attach to the tmux session
tmux attach -t $SESSION_NAME
