
#!/usr/bin/env python3
import asyncio
import aiohttp
import logging
import argparse
import os
import json
import signal
import sys
import psutil
from datetime import datetime, timedelta, timezone
from typing import Any
from tqdm.asyncio import tqdm as tqdm_asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

class RateLimiter:
    """Sliding window rate limiter for requests per minute."""
    
    def __init__(self, rate_per_minute: int):
        self.rate_per_minute = rate_per_minute
        self.window_seconds = 60
        self.request_times: list[float] = []  # Timestamps of requests in current window
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        while True:
            async with self.lock:
                now = datetime.now().timestamp()
                cutoff = now - self.window_seconds
                
                # Remove requests older than 60 seconds
                self.request_times = [t for t in self.request_times if t > cutoff]
                
                # Check if we can make a request
                if len(self.request_times) < self.rate_per_minute:
                    self.request_times.append(now)
                    return
                
                # Calculate wait time until oldest request falls out of window
                oldest = self.request_times[0]
                wait_time = (oldest + self.window_seconds) - now
            
            # Sleep outside the lock
            if wait_time > 0:
                await asyncio.sleep(wait_time + 0.01)  # Add small buffer
            else:
                await asyncio.sleep(0.01)  # Minimal sleep to prevent busy loop

class Statistics:
    """Track ingestion statistics across all workers."""
    
    def __init__(self, history_size: int = 20):
        self.lock = asyncio.Lock()
        self.total_items = 0
        self.total_requests = 0
        self.total_errors = 0
        self.total_rate_limits = 0
        self.completed_ranges = 0
        self.total_ranges = 0
        self.start_time = datetime.now()
        self.last_display_time = datetime.now()
        self.last_items = 0
        self.last_requests = 0
        self.display_task: asyncio.Task | None = None
        self.shutdown_event = asyncio.Event()
        
        # History tracking (per minute)
        self.history_size = history_size
        self.requests_history: list[float] = []
        self.items_history: list[float] = []
        self.memory_history: list[float] = []  # Memory usage in MB
        self.last_chart_lines = 0  # Track lines for clearing
        
        # Get process for memory monitoring
        self.process = psutil.Process()
    
    async def increment_items(self, count: int) -> None:
        # Lock-free for performance - slight inaccuracy acceptable for display stats
        self.total_items += count
    
    async def increment_requests(self) -> None:
        # Lock-free for performance
        self.total_requests += 1
    
    async def increment_errors(self) -> None:
        # Lock-free for performance
        self.total_errors += 1
    
    async def increment_rate_limits(self) -> None:
        # Lock-free for performance
        self.total_rate_limits += 1
    
    async def increment_completed_ranges(self) -> None:
        # Lock-free for performance
        self.completed_ranges += 1
    
    def get_summary(self) -> str:
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        rate = self.total_items / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        return (
            f"\n{'='*60}\n"
            f"INGESTION STATISTICS\n"
            f"{'='*60}\n"
            f"Total items ingested: {self.total_items:,}\n"
            f"Completed ranges: {self.completed_ranges}/{self.total_ranges}\n"
            f"Rate limit hits: {self.total_rate_limits}\n"
            f"Errors encountered: {self.total_errors}\n"
            f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
            f"Average rate: {rate:.1f} items/sec\n"
            f"{'='*60}"
        )
    
    def _create_vertical_chart(self, history: list[float], height: int = 8, width: int = None) -> list[str]:
        """Create a vertical bar chart from history data."""
        if not history:
            return [" " * (width or self.history_size) for _ in range(height)]
        
        chart_width = width or len(history)
        max_val = max(history) if history else 1
        if max_val == 0:
            max_val = 1
        
        # Normalize values to chart height
        normalized = [int((val / max_val) * height) for val in history]
        
        # Build chart from top to bottom
        lines = []
        for row in range(height, 0, -1):
            line = ""
            for val in normalized:
                if val >= row:
                    line += "█"
                else:
                    line += " "
            lines.append(line)
        
        return lines
    
    def get_live_stats(self) -> dict[str, str]:
        """Get compact stats dict for tqdm postfix display."""
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        rate = self.total_items / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        return {
            "total": f"{self.total_items:,}",
            "ranges": f"{self.completed_ranges}/{self.total_ranges}",
            "rate_limits": str(self.total_rate_limits),
            "errors": str(self.total_errors),
            "rate": f"{rate:.1f}/s",
            "time": f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        }
    
    async def display_chart(self, pbar: tqdm_asyncio) -> None:
        """Display current chart state."""
        async with self.lock:
            total_items = self.total_items
            total_requests = self.total_requests
            ranges_done = self.completed_ranges
            ranges_total = self.total_ranges
            rate_limits = self.total_rate_limits
            errors = self.total_errors
            req_history = self.requests_history.copy()
            items_history = self.items_history.copy()
            memory_history = self.memory_history.copy()
        
        # Calculate overall elapsed time
        now = datetime.now()
        total_elapsed = now - self.start_time
        hours, remainder = divmod(int(total_elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Calculate overall rate
        overall_rate = total_items / total_elapsed.total_seconds() if total_elapsed.total_seconds() > 0 else 0
        
        # Get current memory
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Create vertical charts
        chart_height = 8
        req_chart = self._create_vertical_chart(req_history, height=chart_height)
        items_chart = self._create_vertical_chart(items_history, height=chart_height)
        memory_chart = self._create_vertical_chart(memory_history, height=chart_height)
        
        # Get max values for labels
        max_req = max(req_history) if req_history else 0
        max_items = max(items_history) if items_history else 0
        max_memory = max(memory_history) if memory_history else 0
        current_req = req_history[-1] if req_history else 0
        current_items = items_history[-1] if items_history else 0
        
        # Build display
        lines = []
        lines.append(f"{'='*100}")
        lines.append(f"LIVE METRICS | Total: {total_items:,} items ({overall_rate:.1f}/s) | {total_requests:,} requests | Memory: {current_memory:.1f} MB | Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        lines.append(f"{'-'*100}")
        
        # Side-by-side charts with proper alignment
        chart_width = max(len(req_history), len(items_history), len(memory_history), self.history_size)
        spacing = 3  # Space between the charts
        
        # Calculate label widths for proper alignment
        req_label = f"Requests/min (max: {max_req:.0f})"
        items_label = f"Items/min (max: {max_items:.0f})"
        memory_label = f"Memory MB (max: {max_memory:.0f})"
        
        # Align labels
        lines.append(f"  {req_label}{' ' * (chart_width - len(req_label) + spacing)}{items_label}{' ' * (chart_width - len(items_label) + spacing)}{memory_label}")
        
        # Chart bars with consistent spacing
        for i in range(chart_height):
            req_line = req_chart[i].ljust(chart_width)
            items_line = items_chart[i].ljust(chart_width)
            memory_line = memory_chart[i].ljust(chart_width)
            lines.append(f"  {req_line}{' ' * spacing}{items_line}{' ' * spacing}{memory_line}")
        
        # Bottom line and current values
        lines.append(f"  {'─' * chart_width}{' ' * spacing}{'─' * chart_width}{' ' * spacing}{'─' * chart_width}")
        
        current_req_str = f"Current: {current_req:7.0f} req/min"
        current_items_str = f"Current: {current_items:7.0f} items/min"
        current_memory_str = f"Current: {current_memory:7.1f} MB"
        lines.append(f"  {current_req_str}{' ' * (chart_width - len(current_req_str) + spacing)}{current_items_str}{' ' * (chart_width - len(current_items_str) + spacing)}{current_memory_str}")
        lines.append(f"{'-'*100}")
        lines.append(f"Ranges: {ranges_done}/{ranges_total} | Rate Limits: {rate_limits} | Errors: {errors}")
        lines.append(f"{'='*100}")
        
        # Temporarily disable tqdm's dynamic display to avoid conflicts with ANSI codes
        pbar.disable = True
        pbar.clear()
        
        # Clear previous chart if exists
        if self.last_chart_lines > 0:
            # Move cursor up and clear each line
            for _ in range(self.last_chart_lines):
                sys.stdout.write('\033[1A')  # Move up one line
                sys.stdout.write('\033[2K')  # Clear entire line
        
        # Write new chart
        for line in lines:
            sys.stdout.write(line + '\n')
        sys.stdout.flush()
        
        self.last_chart_lines = len(lines)
        
        # Re-enable and refresh tqdm
        pbar.disable = False
        pbar.refresh()
    
    async def _periodic_display(self, pbar: tqdm_asyncio, interval: float = 60.0) -> None:
        """Background task to display live bar charts every minute."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)
                if not self.shutdown_event.is_set():
                    now = datetime.now()
                    elapsed = (now - self.last_display_time).total_seconds()
                    
                    if elapsed > 0:
                        # Calculate rates since last display (per minute)
                        async with self.lock:
                            items_delta = self.total_items - self.last_items
                            requests_delta = self.total_requests - self.last_requests
                            items_per_min = (items_delta / elapsed) * 60
                            requests_per_min = (requests_delta / elapsed) * 60
                            
                            # Get current memory usage in MB
                            memory_mb = self.process.memory_info().rss / 1024 / 1024
                            
                            # Add to history
                            self.requests_history.append(requests_per_min)
                            self.items_history.append(items_per_min)
                            self.memory_history.append(memory_mb)
                            
                            # Keep only recent history
                            if len(self.requests_history) > self.history_size:
                                self.requests_history.pop(0)
                            if len(self.items_history) > self.history_size:
                                self.items_history.pop(0)
                            if len(self.memory_history) > self.history_size:
                                self.memory_history.pop(0)
                            
                            # Update for next iteration
                            self.last_items = self.total_items
                            self.last_requests = self.total_requests
                            self.last_display_time = now
                        
                        # Display updated chart
                        await self.display_chart(pbar)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic stats display: {e}")
    
    async def start_display(self, pbar: tqdm_asyncio, interval: float = 60.0) -> None:
        """Start the background display task (updates every minute)."""
        self.display_task = asyncio.create_task(self._periodic_display(pbar, interval))
    
    async def stop_display(self) -> None:
        """Stop the background display task."""
        self.shutdown_event.set()
        if self.display_task:
            self.display_task.cancel()
            try:
                await self.display_task
            except asyncio.CancelledError:
                pass

class StateManager:
    """Manages resume state with efficient saving on shutdown only."""
    
    def __init__(self, resume_file: str, collection: str):
        self.resume_file = resume_file
        self.collection = collection
        self.state: dict[str, Any] = {}
        self.lock = asyncio.Lock()
    
    async def load_state(self) -> dict[str, Any]:
        """Load state from file."""
        if not self.resume_file or not os.path.exists(self.resume_file):
            return {}
        
        try:
            with open(self.resume_file, "r") as f:
                all_state = json.load(f)
                return all_state.get(self.collection, {})
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not load resume state, starting fresh")
            return {}
    
    async def update_state(self, range_key: str, state_data: dict[str, Any]) -> None:
        """Update state for a specific range (non-blocking)."""
        async with self.lock:
            self.state[range_key] = state_data
    
    async def save_state(self) -> None:
        """Save current state to disk."""
        if not self.resume_file:
            return
        
        async with self.lock:
            # Load existing state to preserve other collections
            all_state = {}
            if os.path.exists(self.resume_file):
                try:
                    with open(self.resume_file, "r") as f:
                        all_state = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            
            # Update state for current collection
            all_state[self.collection] = self.state.copy()
            
            # Write atomically using a temp file
            temp_file = f"{self.resume_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(all_state, f, indent=2)
            os.replace(temp_file, self.resume_file)
    
    async def start(self) -> None:
        """Load initial state."""
        self.state = await self.load_state()
    
    async def stop(self) -> None:
        """Save final state."""
        await self.save_state()
    
    def get_state(self, range_key: str) -> dict[str, Any]:
        """Get state for a specific range."""
        return self.state.get(range_key, {})

async def post_chunk(session: aiohttp.ClientSession, url: str, chunk: list[dict[str, Any]], 
                     semaphore: asyncio.Semaphore, pbar: tqdm_asyncio, worker_id: int, 
                     stats: Statistics, shutdown_event: asyncio.Event,
                     rate_limiter: RateLimiter | None = None) -> int:
    # Prepare JSON payload outside semaphore to avoid blocking other workers
    items_dict = {item['id']: item for item in chunk}
    payload = {"items": items_dict}
    count = len(chunk)
    
    # Retry logic with semaphore release during backoff
    for attempt in range(1, 6):  # 5 attempts
        try:
            # Acquire rate limit token before semaphore
            if rate_limiter:
                await rate_limiter.acquire()
            
            async with semaphore:
                await stats.increment_requests()
                # Add timeout to prevent hanging connections
                timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=60)
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        # Check if it's a rate limit error and track it
                        is_rate_limit = "RateLimitExceeded" in text or resp.status == 429
                        if is_rate_limit:
                            await stats.increment_rate_limits()
                            error_type = "RATE_LIMIT"
                        else:
                            await stats.increment_errors()
                            error_type = "ERROR"
                        
                        # Write error directly to stderr (bypassing tqdm buffer)
                        sys.stderr.write(f"{timestamp} [W{worker_id}] [{error_type}] Bulk POST failed (status {resp.status}): {text[:200]}\n")
                        sys.stderr.flush()
                        raise Exception(f"POST failed with status {resp.status}: {text[:200]}")
                    
                    await stats.increment_items(count)
                    # Simplified progress bar update - just update every chunk
                    if not shutdown_event.is_set():
                        pbar.update(count)
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
            
            # Sleep OUTSIDE semaphore so other workers can proceed
            await asyncio.sleep(wait_time)
    
    return 0  # Should never reach here

async def create_collection_if_needed(args: argparse.Namespace, session: aiohttp.ClientSession) -> None:
    """Create collection on target if it doesn't exist and --create-collection is set."""
    if not args.create_collection:
        return
    
    # Check if collection exists on target
    check_url = f"{args.target}/collections/{args.collection}"
    async with session.get(check_url) as resp:
        if resp.status == 200:
            logging.info(f"Collection {args.collection} already exists on target")
            return
    
    # Fetch collection from source
    logging.info(f"Fetching collection {args.collection} from source...")
    source_url = f"{args.source}/collections/{args.collection}"
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

async def get_collection_temporal_extent(source_url: str, collection_id: str) -> tuple[datetime, datetime]:
    """Fetch collection temporal extent from source."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{source_url}/collections/{collection_id}") as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch collection: {resp.status}")
            collection = await resp.json()
    
    # Get temporal extent
    extent = collection.get("extent", {})
    temporal = extent.get("temporal", {})
    intervals = temporal.get("interval", [])
    
    if intervals and intervals[0]:
        interval = intervals[0]
        start = interval[0]
        end = interval[1]
        
        # Handle None values - ensure all datetimes are UTC aware
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
    
    # Fallback to a reasonable range - use UTC aware datetimes
    return datetime(2000, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc)

def split_date_range(start_date: datetime, end_date: datetime, num_splits: int) -> list[tuple[datetime, datetime]]:
    """Split a date range into num_splits equal chunks."""
    total_days = (end_date - start_date).days
    days_per_split = max(1, total_days // num_splits)
    
    ranges = []
    current = start_date
    
    for i in range(num_splits):
        if i == num_splits - 1:
            # Last split gets everything remaining
            ranges.append((current, end_date))
        else:
            next_date = current + timedelta(days=days_per_split)
            ranges.append((current, next_date))
            current = next_date
    
    return ranges

async def fetch_page_async(session: aiohttp.ClientSession, url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch a search page asynchronously using GET."""
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with session.get(url, params=params, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"Failed to fetch page: {resp.status} - {text[:500]}")
        return await resp.json()

async def fetch_resume_page(next_page_url: str, session: aiohttp.ClientSession) -> dict[str, Any]:
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

def build_state_update(total_posted: int, next_page_url: str | None, 
                      current_item_offset: int, total_items_on_page: int,
                      current_page_url: str | None = None, completed: bool = False) -> dict[str, Any]:
    """Build a state update dict."""
    if completed:
        return {
            "completed": True,
            "next_page_url": None,
            "item_offset": 0,
            "items_posted": total_posted,
            "completed_at": datetime.now().isoformat()
        }
    elif current_item_offset >= total_items_on_page:
        # Finished current page, move to next page with offset 0
        return {
            "completed": False,
            "next_page_url": next_page_url,
            "item_offset": 0,
            "items_posted": total_posted,
            "updated_at": datetime.now().isoformat()
        }
    else:
        # Still processing current page, save current position
        return {
            "completed": False,
            "next_page_url": current_page_url or next_page_url,
            "item_offset": current_item_offset,
            "items_posted": total_posted,
            "updated_at": datetime.now().isoformat()
        }

async def process_page(page: dict[str, Any], page_num: int, item_offset: int, worker_id: int, 
                       args: argparse.Namespace, session: aiohttp.ClientSession, url: str, 
                       semaphore: asyncio.Semaphore, pbar: tqdm_asyncio, 
                       state_manager: StateManager, range_key: str, total_posted: int,
                       current_page_url: str | None, stats: Statistics, 
                       shutdown_event: asyncio.Event, rate_limiter: RateLimiter | None = None) -> tuple[int, bool]:
    """Process a single page of items."""
    items = page.get("features", [])
    if not items:
        return total_posted, False
    
    total_items_on_page = len(items)
    
    # If resuming on first page, skip already processed items
    start_idx = 0
    if page_num == 0 and item_offset > 0:
        start_idx = item_offset
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"{timestamp} [W{worker_id}] Resuming on current page, skipping {item_offset} items")
        items = items[start_idx:]
    
    if not items:
        return total_posted, True
    
    next_page_url = get_next_page_url(page)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    logger.debug(f"{timestamp} [W{worker_id}] Page {page_num}: processing {len(items)} items")
    
    # Split into chunks and post them all in parallel
    chunk_tasks = []
    chunk_positions = []  # Track position of each chunk
    for i in range(0, len(items), args.chunk_size):
        chunk = items[i:i + args.chunk_size]
        chunk_positions.append((i, len(chunk)))
        task = asyncio.create_task(post_chunk(session, url, chunk, semaphore, pbar, worker_id, stats, shutdown_event, rate_limiter))
        chunk_tasks.append(task)
    
    # Wait for all chunks to complete
    results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
    
    # Count successful posts and log errors
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            timestamp = datetime.now().strftime("%H:%M:%S")
            # Write error directly to stderr
            sys.stderr.write(f"{timestamp} [W{worker_id}] [ERROR] Failed to post chunk {idx} after retries: {result}\n")
            sys.stderr.flush()
        else:
            total_posted += result
    
    # Update state only once per page (not per chunk) to reduce lock contention
    state_update = build_state_update(total_posted, next_page_url, 
                                     total_items_on_page, total_items_on_page, current_page_url)
    await state_manager.update_state(range_key, state_update)
    
    return total_posted, True

async def ingest_worker(worker_id: int, args: argparse.Namespace, start_date: datetime, 
                        end_date: datetime, semaphore: asyncio.Semaphore, 
                        pbar: tqdm_asyncio, state_manager: StateManager, 
                        shutdown_event: asyncio.Event, stats: Statistics,
                        session: aiohttp.ClientSession, rate_limiter: RateLimiter | None = None) -> None:
    """Worker to ingest a specific datetime range."""
    range_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
    
    # Check if this range is already completed
    range_state = state_manager.get_state(range_key)
    if range_state.get("completed", False):
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"{timestamp} [W{worker_id}] Range already completed, skipping")
        return
    
    # Get resume position for this range
    items_posted = range_state.get("items_posted", 0)
    next_page_url = range_state.get("next_page_url", None)
    item_offset = range_state.get("item_offset", 0)
    
    if next_page_url:
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"{timestamp} [W{worker_id}] Resuming from saved page ({items_posted} already posted)")
    
    url = f"{args.target}/collections/{args.collection}/bulk_items"
    search_url = f"{args.source}/search"
    
    # Build initial search params
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    search_params = {
        "collections": args.collection,
        "datetime": f"{start_str}/{end_str}",
        "limit": 100
    }
    
    try:
        page_num = 0
        total_posted = items_posted
        current_page_url = next_page_url
        
        # Fetch first page (either resume or initial search)
        if next_page_url:
            page = await fetch_resume_page(next_page_url, session)
        else:
            page = await fetch_page_async(session, search_url, search_params)
        
        # Start prefetching next page
        next_url = get_next_page_url(page)
        next_page_task = asyncio.create_task(fetch_resume_page(next_url, session)) if next_url else None
        
        while page:
            # Check if shutdown was requested
            if shutdown_event.is_set():
                timestamp = datetime.now().strftime("%H:%M:%S")
                logger.debug(f"{timestamp} [W{worker_id}] Shutdown requested, stopping gracefully")
                if next_page_task:
                    next_page_task.cancel()
                # State manager will handle final save
                return
            
            # Process current page while next page is being fetched
            total_posted, should_continue = await process_page(
                page, page_num, item_offset if page_num == 0 else 0,
                worker_id, args, session, url, semaphore, pbar,
                state_manager, range_key, total_posted, current_page_url, stats,
                shutdown_event, rate_limiter
            )
            
            if not should_continue:
                if next_page_task:
                    next_page_task.cancel()
                break
            
            # Wait for prefetched page (should be ready or nearly ready)
            if next_page_task:
                try:
                    page = await next_page_task
                    page_num += 1
                    current_page_url = next_url
                    
                    # Start prefetching the next page
                    next_url = get_next_page_url(page)
                    next_page_task = asyncio.create_task(fetch_resume_page(next_url, session)) if next_url else None
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    sys.stderr.write(f"{timestamp} [W{worker_id}] [ERROR] Failed to fetch next page: {e}\n")
                    sys.stderr.flush()
                    break
            else:
                break
        
        # Mark this range as completed
        state_update = build_state_update(total_posted, None, 0, 0, completed=True)
        await state_manager.update_state(range_key, state_update)
        await stats.increment_completed_ranges()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"{timestamp} [W{worker_id}] Completed range {start_date.date()} to {end_date.date()}: {total_posted} items")
    
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"{timestamp} [W{worker_id}] [ERROR] Failed: {e}\n")
        sys.stderr.flush()
        # State manager will save on shutdown
        raise

async def main(args: argparse.Namespace) -> None:
    logger.info(f"Starting ingestion for collection: {args.collection}")
    
    # Initialize statistics
    stats = Statistics()
    
    # Initialize state manager
    state_manager = StateManager(
        resume_file=args.resume_file or "",
        collection=args.collection
    )
    await state_manager.start()
    
    if state_manager.state:
        logger.info(f"Loaded resume state with {len(state_manager.state)} ranges for collection {args.collection}")
    else:
        logger.info(f"No resume state found for collection {args.collection}, starting fresh")
    
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
            # Force exit on second signal
            loop.stop()
            sys.exit(1)
        else:
            # Already shutting down forcefully
            pass
    
    # Use asyncio-compatible signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    # Create collection once before starting workers
    async with aiohttp.ClientSession() as session:
        await create_collection_if_needed(args, session)
    
    # Determine date range
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        logger.info("No date range specified, fetching from collection metadata...")
        start, end = await get_collection_temporal_extent(args.source, args.collection)
        logger.info(f"Collection temporal extent: {start.date()} to {end.date()}")
    
    # Split range into chunks for parallel processing
    date_ranges = split_date_range(start, end, args.concurrency)
    
    for i, (range_start, range_end) in enumerate(date_ranges):
        logger.debug(f"  Range {i}: {range_start.date()} to {range_end.date()}")

    # Set total ranges for statistics
    stats.total_ranges = len(date_ranges)
    
    # Display initial status using pbar.write() to avoid mixing with progress bar
    logger.info(f"Split date range into {len(date_ranges)} chunks for parallel processing")
    logger.info(f"Starting ingestion with {args.concurrency} workers, chunk size: {args.chunk_size}")
    logger.info(f"Semaphore limit: {args.concurrency * 10} concurrent POSTs")
    logger.info(f"Connection pool: {args.concurrency * 10} total, {args.concurrency * 8} per host")
    logger.info(f"Statistics will be displayed every 30 seconds")

    # Create shared semaphore and progress bar
    semaphore = asyncio.Semaphore(args.concurrency * 10)  # Allow more POST concurrency for bulk operations
    
    # Create global rate limiter if specified
    rate_limiter = RateLimiter(args.rate_limit) if args.rate_limit else None
    if rate_limiter:
        logger.info(f"Rate limiting enabled: {args.rate_limit} requests per minute")

    # Enable progress bar with live statistics
    pbar = tqdm_asyncio(total=0, unit=" items", desc="Ingestion Progress", 
                       disable=False, leave=True, mininterval=1.0, position=0)
    
    # Create shared session with optimized connection pooling for all workers
    connector = aiohttp.TCPConnector(
        limit=args.concurrency * 10,
        limit_per_host=args.concurrency * 8,
        ttl_dns_cache=300  # Cache DNS for 5 minutes
    )
    
    try:
        async with aiohttp.ClientSession(connector=connector) as shared_session:
            # Start live statistics display (updates every 30 seconds with per-minute rates)
            await stats.start_display(pbar, interval=30.0)
            
            # Display initial chart
            await stats.display_chart(pbar)
            
            # Create worker tasks - all share the same session
            worker_tasks = [
                ingest_worker(i, args, range_start, range_end, semaphore, pbar, state_manager, shutdown_event, stats, shared_session, rate_limiter)
                for i, (range_start, range_end) in enumerate(date_ranges)
            ]
            
            # Run all workers concurrently
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            
            if shutdown_event.is_set():
                logger.info("Shutdown completed gracefully")
            else:
                logger.info(f"✓ Ingestion complete! All {len(date_ranges)} ranges processed.")
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received, shutting down...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise
    finally:
        pbar.close()
        # Stop live statistics display
        await stats.stop_display()
        
        # Stop state manager (does final save)
        logger.info("Saving final state...")
        await state_manager.stop()
        if args.resume_file:
            logger.info(f"Resume state saved to {args.resume_file}")
        
        # Display final statistics
        print(stats.get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAC ingestion with concurrency, progress, and resume")
    parser.add_argument("--source", required=True, help="Source STAC API URL")
    parser.add_argument("--target", required=True, help="Target STAC API URL")
    parser.add_argument("--collection", required=True, help="Collection ID to ingest")
    parser.add_argument("--chunk-size", type=int, default=500, help="Items per bulk POST")
    parser.add_argument("--concurrency", type=int, default=50, help="Parallel POST requests")
    parser.add_argument("--rate-limit", type=int, help="Maximum requests per minute (global limit)")
    parser.add_argument("--resume-file", help="Path to resume state file")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD) for chunking")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD) for chunking")
    parser.add_argument("--create-collection", action="store_true", help="Create collection on target if it doesn't exist")
    args = parser.parse_args()

    asyncio.run(main(args))
