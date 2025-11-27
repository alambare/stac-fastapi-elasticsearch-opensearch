#!/usr/bin/env python3
from dataclasses import dataclass
import asyncio
import traceback
import aiohttp
import sys
import argparse
import signal
from tqdm.asyncio import tqdm as tqdm_asyncio

# Needed for type hints and logic
from typing import Any
from datetime import datetime, timezone, timedelta
import logging

# Import modularized components
from rate_limiter import RateLimiter, read_rate_limited
from compute_healpix import compute_healpix_ids
from stats import Statistics
from state_manager import StateManager, build_state_update

from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

logger.setLevel(logging.WARNING)


@dataclass
class IngestConfig:
    source: str
    target: str
    collection: str
    chunk_size: int
    concurrency: int
    min_level: int 
    max_level: int
    push_rate_limit: int | None = None
    read_rate_limit: int | None = None
    resume_file: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    create_collection: bool = False
    max_workers: int | None = None

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'IngestConfig':
        return IngestConfig(
            source=args.source,
            target=args.target,
            collection=args.collection,
            chunk_size=args.chunk_size,
            concurrency=args.concurrency,
            push_rate_limit=args.push_rate_limit,
            read_rate_limit=args.read_rate_limit,
            resume_file=args.resume_file,
            start_date=args.start_date,
            end_date=args.end_date,
            create_collection=args.create_collection,
            min_level=args.min_level,
            max_level=args.max_level,
            max_workers=args.max_workers,
        )


@dataclass
class IngestRuntime:
    """Group runtime objects for ingest operations."""
    session: aiohttp.ClientSession
    semaphore: asyncio.Semaphore
    pbar: tqdm_asyncio
    state_manager: StateManager
    stats: Statistics
    shutdown_event: asyncio.Event
    push_rate_limiter: RateLimiter | None = None
    read_rate_limiter: RateLimiter | None = None
    executor: ProcessPoolExecutor | None = None

async def get_collection_temporal_extent(source_url: str, collection_id: str) -> tuple[datetime, datetime]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{source_url}/collections/{collection_id}") as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch collection: {resp.status}")
            collection = await resp.json()
    extent = collection.get("extent", {})
    temporal = extent.get("temporal", {})
    intervals = temporal.get("interval", [])
    if intervals and intervals[0]:
        interval = intervals[0]
        start = interval[0]
        end = interval[1]
        if start is None:
            start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        elif isinstance(start, str):
            start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        elif isinstance(start, datetime) and start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)
        elif isinstance(end, str):
            end = datetime.fromisoformat(end.replace('Z', '+00:00'))
        elif isinstance(end, datetime) and end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return start, end
    return datetime(2000, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc)

def split_date_range(start_date: datetime, end_date: datetime, num_splits: int) -> list[tuple[datetime, datetime]]:
    total_days = (end_date - start_date).days
    days_per_split = max(1, total_days // num_splits)
    ranges = []
    current = start_date
    for i in range(num_splits):
        if i == num_splits - 1:
            ranges.append((current, end_date))
        else:
            next_date = current + timedelta(days=days_per_split)
            ranges.append((current, next_date))
            current = next_date
    return ranges

def compute_healpix_batch(args_list):
    # args_list: list of (geometry, min_level, max_level)
    return [compute_healpix_ids(geom, min_level, max_level) for geom, min_level, max_level in args_list]

async def post_chunk(
    runtime: IngestRuntime,
    url: str,
    chunk: list[dict[str, Any]],
    worker_id: int,
) -> int:
    """Post a chunk of items with retry and rate limiting."""

    items_dict = {item['id']: item for item in chunk}
    payload = {"items": items_dict}
    count = len(chunk)

    # Retry logic with semaphore release during backoff
    import time
    for attempt in range(1, 6):  # 5 attempts
        try:
            # Time spent waiting for rate limiter
            rate_start = time.perf_counter()
            if runtime.push_rate_limiter:
                await runtime.push_rate_limiter.acquire()
            rate_end = time.perf_counter()
            if rate_end - rate_start > 0.01:
                logger.info(f"[W{worker_id}] Waited {rate_end - rate_start:.3f}s for push rate limiter")

            # Time spent waiting for semaphore and POST
            sem_start = time.perf_counter()
            async with runtime.semaphore:
                sem_acquired = time.perf_counter()
                await runtime.stats.increment_requests()
                # Add timeout to prevent hanging connections
                timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=60)
                post_start = time.perf_counter()
                async with runtime.session.post(url, json=payload, timeout=timeout) as resp:
                    post_end = time.perf_counter()
                    if resp.status != 200:
                        text = await resp.text()
                        timestamp = datetime.now().strftime("%H:%M:%S")

                        is_rate_limit = "RateLimitExceeded" in text or resp.status == 429
                        if is_rate_limit:
                            await runtime.stats.increment_rate_limits()
                            error_type = "RATE_LIMIT"
                        else:
                            await runtime.stats.increment_errors()
                            error_type = "ERROR"

                        sys.stderr.write(f"{timestamp} [W{worker_id}] [{error_type}] Bulk POST failed (status {resp.status}): {text[:200]}\n")
                        sys.stderr.flush()
                        raise Exception(f"POST failed with status {resp.status}: {text[:200]}")

                    await runtime.stats.increment_items(count)
                    # Simplified progress bar update - just update every chunk
                    if not runtime.shutdown_event.is_set():
                        runtime.pbar.update(count)
                    logger.info(f"[W{worker_id}] POST chunk ({count} items): rate_wait={rate_end-rate_start:.3f}s, sem_wait={sem_acquired-sem_start:.3f}s, post_time={post_end-post_start:.3f}s")
                    return count
        except Exception as e:
            if attempt == 5:
                raise

            # Calculate backoff time outside semaphore
            error_msg = str(e)
            is_rate_limit = "RateLimitExceeded" in error_msg or "429" in error_msg

            if is_rate_limit:
                wait_time = min(60, 10 * attempt)  # 10s, 20s, 30s, 40s, max 60s
            else:
                wait_time = 0.5 * attempt  # Regular exponential backoff

            logger.info(f"[W{worker_id}] Backing off for {wait_time:.2f}s after error: {e}")
            # Sleep OUTSIDE semaphore so other workers can proceed
            await asyncio.sleep(wait_time)

    logger.error(f"[W{worker_id}] POST chunk failed after 5 attempts")
    return 0  # Should never reach here


async def create_collection_if_needed(config: IngestConfig, session: aiohttp.ClientSession) -> None:
    """Create collection on target if it doesn't exist and --create-collection is set."""
    if not config.create_collection:
        return
    
    # Check if collection exists on target
    check_url = f"{config.target}/collections/{config.collection}"
    async with session.get(check_url) as resp:
        if resp.status == 200:
            logging.info(f"Collection {config.collection} already exists on target")
            return
    
    # Fetch collection from source
    logging.info(f"Fetching collection {config.collection} from source...")
    source_url = f"{config.source}/collections/{config.collection}"
    async with session.get(source_url) as resp:
        if resp.status != 200:
            text = await resp.text()
            logging.error(f"Failed to fetch collection from source: {text[:500]}")
            raise Exception(f"Cannot fetch collection from source (status {resp.status})")
        collection_data = await resp.json()
    
    # Remove read-only fields that might cause issues
    # Keep links but create minimal valid structure if needed
    if not collection_data.get('links'):
        collection_data['links'] = []
    
    # Remove other potentially problematic fields
    for field in ['created', 'updated', 'conformsTo']:
        collection_data.pop(field, None)
    
    # Create collection on target
    logging.info(f"Creating collection {args.collection} on target...")
    create_url = f"{args.target}/collections"
    async with session.post(create_url, json=collection_data) as resp:
        if resp.status not in (200, 201):
            text = await resp.text()
            logging.error(f"Failed to create collection (full error): {text}")
            raise Exception(f"Cannot create collection (status {resp.status})")
        logging.info(f"Collection {args.collection} created successfully")


@read_rate_limited
async def fetch_page_async(session: aiohttp.ClientSession, url: str, params: dict[str, Any], source_rate_limiter=None) -> dict[str, Any]:
    """Fetch a search page asynchronously using GET, with retry logic."""
    import asyncio
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Failed to fetch page: {resp.status} - {text[:500]}")
                return await resp.json()
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait_time = min(30, 2 ** attempt)
            logger.warning(f"fetch_page_async: attempt {attempt} failed with error: {e}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

@read_rate_limited
async def fetch_resume_page(next_page_url: str, session: aiohttp.ClientSession, source_rate_limiter=None) -> dict[str, Any]:
    """Fetch a page using its URL."""
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with session.get(next_page_url, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"Failed to fetch resume page: {resp.status} - {text[:500]}")
        return await resp.json()

def get_next_page_url(page: dict[str, Any]) -> str | None:
    """Extract next page URL from page links."""
    for link in page.get("links", []):
        if link.get("rel") == "next":
            return link.get("href")
    return None





async def process_page(
    page: dict[str, Any],
    page_num: int,
    item_offset: int,
    worker_id: int,
    config: IngestConfig,
    runtime: IngestRuntime,
    url: str,
    range_key: str,
    total_posted: int,
    current_page_url: str | None
) -> tuple[int, bool]:
    """Process a single page of items."""
    items = page.get("features", [])
    if not items:
        return total_posted, False
    # If resuming on first page, skip already processed items
    if page_num == 0 and item_offset > 0:
        items = items[item_offset:]
    next_page_url = get_next_page_url(page)
    # Just return the items and next_page_url for aggregation in ingest_worker
    return items, next_page_url

async def ingest_worker(
        worker_id: int,
        config: IngestConfig,
        runtime: IngestRuntime,
        start_date: datetime, 
        end_date: datetime
    ) -> None:

    """Worker to ingest a specific datetime range."""
    range_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
    
    # Check if this range is already completed
    range_state = runtime.state_manager.get_state(range_key)
    if range_state.get("completed", False):
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"{timestamp} [W{worker_id}] Range already completed, skipping")
        return
    
    # Get resume position for this range
    items_posted = range_state.get("items_posted", 0)
    next_page_url = range_state.get("next_page_url", None)
    item_offset = range_state.get("item_offset", 0)

    # Initialize variables for page tracking
    page_num = 0
    current_page_url = next_page_url if next_page_url else None
    total_posted = items_posted

    if next_page_url:
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"{timestamp} [W{worker_id}] Resuming from saved page ({items_posted} already posted)")

    url = f"{config.target}/collections/{config.collection}/bulk_items"
    search_url = f"{config.source}/search"

    # Build initial search params
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    search_params = {
        "collections": config.collection,
        "datetime": f"{start_str}/{end_str}",
        "limit": 100
    }

    import time
    logger.info(f"[W{worker_id}] Worker started for range {start_date} to {end_date}")
    try:
        # Buffer to aggregate items across pages
        item_buffer = []

        page = None
        if next_page_url:
            fetch_start = time.perf_counter()
            page = await fetch_resume_page(next_page_url, runtime.session, source_rate_limiter=runtime.read_rate_limiter)
            fetch_end = time.perf_counter()
            logger.info(f"[W{worker_id}] First page fetch: {fetch_end - fetch_start:.3f}s")
        else:
            fetch_start = time.perf_counter()
            page = await fetch_page_async(runtime.session, search_url, search_params, source_rate_limiter=runtime.read_rate_limiter)
            fetch_end = time.perf_counter()
            logger.info(f"[W{worker_id}] First page fetch: {fetch_end - fetch_start:.3f}s")

        next_url = get_next_page_url(page)
        next_page_task = asyncio.create_task(fetch_resume_page(next_url, runtime.session, source_rate_limiter=runtime.read_rate_limiter)) if next_url else None
        while page:
            loop_start = time.perf_counter()
            if runtime.shutdown_event.is_set():
                timestamp = datetime.now().strftime("%H:%M:%S")
                logger.debug(f"{timestamp} [W{worker_id}] Shutdown requested, stopping gracefully")
                if next_page_task:
                    next_page_task.cancel()
                return

            # Add items from this page to buffer (respect item_offset for first page)
            items = page.get("features", [])
            if page_num == 0 and item_offset > 0:
                items = items[item_offset:]
            item_buffer.extend(items)

            # While buffer has enough for a chunk, process a chunk
            while len(item_buffer) >= config.chunk_size:
                chunk = item_buffer[:config.chunk_size]
                item_buffer = item_buffer[config.chunk_size:]
                # Compute HEALPix in parallel for chunk
                compute_args = [(item['geometry'], config.min_level, config.max_level) for item in chunk]
                loop = asyncio.get_running_loop()
                hp_start = time.perf_counter()
                healpix_results = await loop.run_in_executor(
                    runtime.executor,
                    compute_healpix_batch,
                    compute_args
                )
                hp_end = time.perf_counter()
                for item, healpix in zip(chunk, healpix_results):
                    if 'properties' not in item:
                        item['properties'] = {}
                    item['properties']['dggs:healpix_nested'] = healpix
                post_start = time.perf_counter()
                await post_chunk(runtime, url, chunk, worker_id)
                post_end = time.perf_counter()
                logger.info(f"[W{worker_id}] Aggregated POST chunk ({len(chunk)} items): HEALPix {hp_end-hp_start:.3f}s, POST {post_end-post_start:.3f}s, loop total {post_end-loop_start:.3f}s")
                total_posted += len(chunk)
                loop_start = time.perf_counter()  # reset for next chunk

            # Prefetch next page
            if next_page_task:
                try:
                    fetch_start = time.perf_counter()
                    page = await next_page_task
                    fetch_end = time.perf_counter()
                    logger.info(f"[W{worker_id}] Next page fetch: {fetch_end - fetch_start:.3f}s")
                    page_num += 1
                    next_url = get_next_page_url(page)
                    next_page_task = asyncio.create_task(fetch_resume_page(next_url, runtime.session, source_rate_limiter=runtime.read_rate_limiter)) if next_url else None
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    sys.stderr.write(f"{timestamp} [W{worker_id}] [ERROR] Failed to fetch next page: {e}\n")
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    break
            else:
                break

        # After all pages, process any remaining items in buffer
        if item_buffer:
            chunk_start = time.perf_counter()
            compute_args = [(item['geometry'], config.min_level, config.max_level) for item in item_buffer]
            loop = asyncio.get_running_loop()
            hp_start = time.perf_counter()
            healpix_results = await loop.run_in_executor(
                runtime.executor,
                compute_healpix_batch,
                compute_args
            )
            hp_end = time.perf_counter()
            for item, healpix in zip(item_buffer, healpix_results):
                if 'properties' not in item:
                    item['properties'] = {}
                item['properties']['dggs:healpix_nested'] = healpix
            post_start = time.perf_counter()
            await post_chunk(runtime, url, item_buffer, worker_id)
            post_end = time.perf_counter()
            logger.info(f"[W{worker_id}] Aggregated POST chunk (final {len(item_buffer)} items): HEALPix {hp_end-hp_start:.3f}s, POST {post_end-post_start:.3f}s, chunk total {post_end-chunk_start:.3f}s")
            total_posted += len(item_buffer)

        # Mark this range as completed
        state_update = build_state_update(total_posted, None, 0, 0, completed=True)
        await runtime.state_manager.update_state(range_key, state_update)
        await runtime.stats.increment_completed_ranges()

        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.info(f"[W{worker_id}] Completed range {start_date.date()} to {end_date.date()}: {total_posted} items")

    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"{timestamp} [W{worker_id}] [ERROR] Failed: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise

async def main(args: argparse.Namespace) -> None:

    config = IngestConfig.from_args(args)

    logger.warning(f"[CONFIG] Using chunk_size: {config.chunk_size}")
    logger.info(f"Starting ingestion for collection: {config.collection}")
    
    stats = Statistics()
    
    state_manager = StateManager(
        resume_file=config.resume_file or "",
        collection=config.collection
    )
    await state_manager.start()
    
    if state_manager.state:
        logger.info(f"Loaded resume state with {len(state_manager.state)} ranges for collection {config.collection}")
    else:
        logger.info(f"No resume state found for collection {config.collection}, starting fresh")

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    shutdown_count = 0

    def signal_handler():
        nonlocal shutdown_count
        shutdown_count += 1
        if shutdown_count == 1:
            logger.info("\nReceived shutdown signal, initiating graceful shutdown...")
            shutdown_event.set()
        elif shutdown_count == 2:
            logger.warning("\nReceived second signal, forcing immediate shutdown...")
            loop.stop()
            sys.exit(1)
        else:
            # Already shutting down forcefully
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Create collection once before starting workers
    async with aiohttp.ClientSession() as session:
        await create_collection_if_needed(args, session)

    if config.start_date and config.end_date:
        start = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        logger.info("No date range specified, fetching from collection metadata...")
        start, end = await get_collection_temporal_extent(config.source, config.collection)
        logger.info(f"Collection temporal extent: {start.date()} to {end.date()}")


    date_ranges = split_date_range(start, end, config.concurrency)

    # ...existing code...

    # Set total ranges for statistics
    stats.total_ranges = len(date_ranges)

    # Display initial status using pbar.write() to avoid mixing with progress bar
    logger.info(f"Starting ingestion with {config.concurrency} workers, chunk size: {config.chunk_size}")
    logger.info(f"Semaphore limit: {config.concurrency * 10} concurrent POSTs")
    logger.info(f"Connection pool: {config.concurrency * 10} total, {config.concurrency * 8} per host")
    logger.info("Statistics will be displayed every 30 seconds")

    semaphore = asyncio.Semaphore(config.concurrency * 10)

    read_rate_limiter = RateLimiter(config.read_rate_limit) if config.read_rate_limit else None
    if read_rate_limiter:
        logger.info(f"Read rate limiting enabled: {config.read_rate_limit} requests per minute")

    push_rate_limiter = RateLimiter(config.push_rate_limit) if config.push_rate_limit else None
    if push_rate_limiter:
        logger.info(f"Push rate limiting enabled: {config.push_rate_limit} requests per minute")

    pbar = tqdm_asyncio(total=0, unit=" items", desc="Ingestion Progress", 
                       disable=False, leave=True, mininterval=1.0, position=0)
    
    connector = aiohttp.TCPConnector(
            limit=config.concurrency * 10,
            limit_per_host=config.concurrency * 8,
            ttl_dns_cache=300  # Cache DNS for 5 minutes
        )

    try:
        async with aiohttp.ClientSession(connector=connector) as shared_session:
            await stats.start_display(pbar, interval=30.0)

            await stats.display_chart(pbar)

            executor = ProcessPoolExecutor(max_workers=config.max_workers) if config.max_workers > 1 else None

            runtime = IngestRuntime(
                session=shared_session,
                semaphore=semaphore,
                pbar=pbar,
                state_manager=state_manager,
                stats=stats,
                shutdown_event=shutdown_event,
                push_rate_limiter=push_rate_limiter,
                read_rate_limiter=read_rate_limiter,
                executor=executor
            )

            worker_tasks = [
                ingest_worker(i, config, runtime, range_start, range_end,)
                for i, (range_start, range_end) in enumerate(date_ranges)
            ]
            await asyncio.gather(*worker_tasks, return_exceptions=True)

            if shutdown_event.is_set():
                logger.info("Shutdown completed gracefully")
            else:
                logger.info(f"Ingestion complete! All {len(date_ranges)} ranges processed.")
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received, shutting down...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise
    finally:
        pbar.close()
        await stats.stop_display()
        logger.info("Saving final state...")
        await state_manager.stop()
        if config.resume_file:
            logger.info(f"Resume state saved to {config.resume_file}")
        print(stats.get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAC ingestion with concurrency, progress, and resume")
    parser.add_argument("--source", required=True, help="Source STAC API URL")
    parser.add_argument("--target", required=True, help="Target STAC API URL")
    parser.add_argument("--collection", required=True, help="Collection ID to ingest")
    parser.add_argument("--chunk-size", type=int, default=100, help="Items per bulk POST")
    parser.add_argument("--concurrency", type=int, default=50, help="Parallel POST requests")
    parser.add_argument("--push-rate-limit", type=int, help="Push maximum requests per minute (global limit)")
    parser.add_argument("--read-rate-limit", type=int, help="Read maximum requests per minute (global limit)")
    parser.add_argument("--resume-file", help="Path to resume state file")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD) for chunking")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD) for chunking")
    parser.add_argument("--create-collection", action="store_true", help="Create collection on target if it doesn't exist")
    parser.add_argument("--min-level", type=int, default=1, help="Minimum HEALPix level (default: 1)")
    parser.add_argument("--max-level", type=int, default=11, help="Maximum HEALPix level (default: 11)")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker processes for HEALPix computation")
    args = parser.parse_args()

    asyncio.run(main(args))
