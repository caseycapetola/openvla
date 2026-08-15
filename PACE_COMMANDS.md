# PACE Commands

## Request a head node with a GPU

```bash
salloc -A gts-szonouz6 -q embers -N1 --ntasks-per-node=8 -t1:00:00 --gres=gpu:A100:1
```

## Ping Openvla-OFT Server

```bash
curl -X POST http://localhost:8777/act -d '{}'
```
