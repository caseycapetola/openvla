#!/usr/bin/env bash
salloc -A gts-szonouz6 -q embers -N1 --ntasks-per-node=2 -t0:20:00 --gres=gpu:A100:1 --gpus=A100:1
