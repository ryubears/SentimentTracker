# Running the tracker on AWS

One Lambda, triggered by EventBridge, holding the SQLite database in S3 between
runs. The pipeline code is unchanged: `lambda_handler` pulls the database, calls
the same `run_period.main()` a local run calls, and pushes it back.

## Why S3 and not EFS

Mounting EFS forces the function into a VPC, and a VPC Lambda needs a NAT
Gateway to reach the X, Binance, Anthropic and Coinbase APIs — about **$32/month
of fixed cost** before a single invocation. The database is ~4 MB and there is
exactly one writer firing once an hour, so copying it in and out of S3 keeps the
function outside a VPC and the bill in pennies.

Two guards keep that safe rather than merely convenient:

- `ReservedConcurrentExecutions: 1` — AWS will not start a second run while one
  is in flight.
- `push()` sends the object's ETag as `If-Match`, so if something *did* write in
  between, the upload fails loudly instead of silently discarding the other
  run's periods.

If the workload ever grows a second writer, that is the point to move to RDS —
not before.

## Deploy

```bash
# 1. Secrets. Keys live only here; nothing is baked into the image.
aws secretsmanager create-secret --name sentiment-tracker/api-keys \
  --secret-string '{"X_BEARER_TOKEN":"...","ANTHROPIC_API_KEY":"..."}'

# 2. Build and deploy (SAM builds the container image and pushes it to ECR).
sam build && sam deploy --guided

# 3. Seed the database, so the first scheduled run has history to resolve against.
#    Run the backfill locally, then upload the result to the bucket SAM created.
/usr/bin/python3 backfill.py --days 60
aws s3 cp data/tracker.sqlite s3://<BucketName from stack outputs>/tracker.sqlite
```

`ScheduleExpression` defaults to `cron(0 * * * ? *)` (hourly, matching the `1h`
horizon). For a daily setup use `cron(0 0 * * ? *)` and set `horizon: "1d"` in
`config.yaml`.

## Operating it

```bash
sam logs -n TrackerFunction --stack-name <stack> --tail   # watch a run
aws s3 cp s3://<bucket>/tracker.sqlite data/tracker.sqlite  # pull state down
/usr/bin/python3 dashboard.py                               # then inspect locally
```

The bucket is versioned, so a bad run can be rolled back to the previous
database object. A `FailureAlarm` fires on any Lambda error — a failed run
otherwise leaves no period behind and nothing else notices. Point it at an SNS
topic to actually get told.

## Environment variables

All optional; unset means "behave exactly as before", so local runs are
unaffected by any of this.

| Variable | Purpose |
|---|---|
| `TRACKER_CONFIG` | path to `config.yaml` (Lambda: `/var/task/config.yaml`) |
| `TRACKER_DB_PATH` | overrides `db_path` (Lambda: `/tmp/tracker.sqlite`) |
| `TRACKER_SECRET_ID` | Secrets Manager secret whose JSON keys become env vars |
| `TRACKER_S3_BUCKET` | bucket holding the database; unset disables S3 sync |
| `TRACKER_S3_KEY` | object key, default `tracker.sqlite` |

## Trading

`trading.enabled` is `false` in the committed config, so a deployed run records
hold/dry-run decisions and sends no orders. Going live means editing
`config.yaml`, rebuilding the image, and adding `COINBASE_API_KEY` /
`COINBASE_API_SECRET` to the secret — deliberately not a runtime toggle.
