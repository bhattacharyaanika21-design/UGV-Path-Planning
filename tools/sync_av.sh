#!/usr/bin/env bash
# The ROS 2 package ships its own copy of the av/ library so it can be built and
# installed on its own. Run this after changing anything in av/ to keep them identical.
set -euo pipefail
cd "$(dirname "$0")/.."
dst=ros2_ws/src/bharat_sim/bharat_sim/av
rm -rf "$dst"
cp -r av "$dst"
find "$dst" -name __pycache__ -type d -prune -exec rm -rf {} +    
echo "synced av/ -> $dst" 
