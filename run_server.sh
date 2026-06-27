#!/usr/bin/env bash
python vla-scripts/deploy.py \
	--openvla_path openvla/openvla-7b \
	--host 0.0.0.0 \
	--port 8000
