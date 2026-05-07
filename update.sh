#!/bin/bash
cd ~/notebook-jupyter
docker compose down
git checkout -- .
git pull origin main
docker compose up -d
echo "Updated and restarted"
