# VSDS Bảng C - Round 2 Submission Kit

Bộ nộp bài gồm Docker, inference pipeline, self-consistency N=5, choice shuffle và audit votes.

## Chạy nhanh
```bash
docker build -t medqa-ensemble:final .
docker run --gpus all --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs medqa-ensemble:final infer --input /app/data/public_test.json --output /app/outputs/submission.csv
```

## File đầu ra
- `submission.csv`: qid,answer
- `votes_audit.json`: log vote counts, seed, run_id, config để audit/tái lập

## Verify reproducibility
```bash
docker run --gpus all --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs medqa-ensemble:final verify_repro --audit /app/outputs/votes_audit.json --input /app/data/public_test.json
```
