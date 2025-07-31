#!/bin/bash

# Configuration
SERVICE_NAME="evqa_orchestrator"
USERNAME="ubuntu"
PROJECT_DIR="/path/to/ev-charging-qa-pipeline"
VENV_PATH="/path/to/venv/bin/python"
LOG_DIR="/var/log/evqa"

# Create log directory
sudo mkdir -p $LOG_DIR
sudo chown $USERNAME:$USERNAME $LOG_DIR

# Create service file
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
sudo tee $SERVICE_FILE > /dev/null <<EOL
[Unit]
Description=EVQA Pipeline Orchestrator
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_PATH src/orchestration/workflow.py
Restart=always
RestartSec=30
StandardOutput=file:$LOG_DIR/orchestrator.out
StandardError=file:$LOG_DIR/orchestrator.err
Environment="PYTHONPATH=$PROJECT_DIR"

[Install]
WantedBy=multi-user.target
EOL

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

echo "Service created: $SERVICE_FILE"
echo "To start: sudo systemctl start $SERVICE_NAME"