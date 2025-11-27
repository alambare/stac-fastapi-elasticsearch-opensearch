import asyncio
from datetime import datetime
import logging
import sys
import psutil
from tqdm.asyncio import tqdm as tqdm_asyncio
logger = logging.getLogger(__name__)

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
        try:
            while not self.shutdown_event.is_set():
                await self.display_chart(pbar)
                await asyncio.sleep(interval)
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
