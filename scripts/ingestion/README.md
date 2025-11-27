## Ingest STAC items into the API from a remote STAC API

The script will compute healpix index for levels from `1` to `11` and ingest them into the items' properties as described below.

### Usage example

> Run `python scripts/ingestion/ingest_stac.py --help` for the full options description

```bash
# start the services
docker compose up opensearch app-opensearch

# harvest
python scripts/ingestion/ingest_stac.py --source https://earth-search.aws.element84.com/v1 --target http://localhost:8082 --collection sentinel-2-l2a --resume-file resume.txt --max-workers 10
```

### Some ingestion statistics

#### Script statistics

```
LIVE METRICS | Total: 8,611,091 items (298.2/s) | 86,152 requests | Memory: 1371.2 MB | Time: 08:01:15

Total items ingested: 8,617,791
Completed ranges: 10/50
Rate limit hits: 0
Errors encountered: 0
Elapsed time: 08:01:34
Average rate: 298.2 items/sec
```

#### Opensearch index statistic for the ingested collection.

```
health status index                                                    uuid                   pri rep docs.count docs.deleted store.size pri.store.size
yellow open   items_sentinel-2-l2a_73656e74696e656c2d322d6c3261-000001 -IlzR4MGQ3a7YQTYPJopxg   1   1    7978291       585243    196.3gb        196.3gb
```