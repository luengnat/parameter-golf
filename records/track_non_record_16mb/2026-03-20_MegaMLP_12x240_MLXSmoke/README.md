This folder captures a non-record MLX smoke experiment for the `megamlp_12x240` candidate.

Measured result:
- `final_int8_zlib_roundtrip_exact val_bpb:2.93614063`
- `final_int8_zlib_roundtrip_exact val_loss:4.96590042`

Configuration:
- `NUM_LAYERS=12`
- `MODEL_DIM=240`
- `NUM_HEADS=6`
- `NUM_KV_HEADS=3`
- `MLP_MULT=8`
- `TIE_EMBEDDINGS=1`
- `VOCAB_SIZE=1024`
- `TRAIN_SEQ_LEN=1024`

Notes:
- This folder is intentionally minimal and starts from `origin/main`.
- It is a hypothesis test, not a leaderboard submission.
- The GPU `~1.17 val_bpb` figure remains a projection, not a measured result.
- The authoritative run log is included as [`train.log`](./train.log).

Reproduction:
```bash
RUN_ID=innov_megamlp_12x240 \
NUM_LAYERS=12 \
MODEL_DIM=240 \
NUM_HEADS=6 \
NUM_KV_HEADS=3 \
MLP_MULT=8 \
ITERATIONS=20 \
WARMUP_STEPS=1 \
VAL_LOSS_EVERY=0 \
VAL_MAX_BATCHES=1 \
TRAIN_BATCH_TOKENS=16384 \
TRAIN_SEQ_LEN=1024 \
python3 train_gpt_mlx.py
```
