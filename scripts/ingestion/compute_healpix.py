
"""
CLI tool to compute and index HEALPix IDs for STAC items in Elasticsearch/OpenSearch.

This script processes STAC items from an Elasticsearch or OpenSearch index, computes 
HEALPix identifiers at multiple levels based on full geometry intersection (not centroids),
and updates the items with the computed data using bulk operations.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import aiohttp
import cdshealpix
import numpy as np
from astropy.coordinates import Longitude, Latitude
import astropy.units as u
from shapely.geometry import shape
from shapely.ops import unary_union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_healpix_cells_for_geometry(
    geometry: dict[str, Any], 
    level: int,
    max_cells: int = 100000
) -> np.ndarray:
    """
    Get all HEALPix cells that intersect with a geometry at a given level.
    
    Optimized version using NumPy arrays throughout and early exit for huge geometries.
    
    Args:
        geometry: GeoJSON geometry dictionary
        level: HEALPix level (0-29)
        max_cells: Maximum number of cells to return (safety limit)
    
    Returns:
        NumPy array of unique HEALPix cell IDs (sorted)
    """
    if level < 0 or level > 29:
        raise ValueError(f"HEALPix level must be between 0 and 29, got {level}")
    
    geom = shape(geometry)
    nside = 2 ** level
    
    # Fast path for points
    if geom.geom_type == 'Point':
        pix_id = cdshealpix.lonlat_to_healpix(
            np.array([geom.x]), 
            np.array([geom.y]), 
            nside, 
            nest=True
        )
        return pix_id.astype(np.int64)
    
    # Get exterior coordinates efficiently
    if geom.geom_type == 'Polygon':
        coords = np.array(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        coords = np.array(unary_union(geom.geoms).exterior.coords)
    else:
        coords = np.array(geom.convex_hull.exterior.coords)
    
    # Extract lon/lat as contiguous arrays
    lons = np.ascontiguousarray(coords[:, 0])
    lats = np.ascontiguousarray(coords[:, 1])
    
    # Convert to astropy objects
    lon_astropy = Longitude(lons, unit=u.deg)
    lat_astropy = Latitude(lats, unit=u.deg)
    
    # Compute HEALPix cells
    ipix, _, _ = cdshealpix.polygon_search(lon_astropy, lat_astropy, level, flat=True)
    
    # Safety check
    if len(ipix) > max_cells:
        logger.warning(f"Geometry produces {len(ipix)} cells at level {level}, truncating to {max_cells}")
        ipix = ipix[:max_cells]
    
    # Return unique sorted cells as int64
    return np.unique(ipix).astype(np.int64)


def compute_healpix_ids(
    geometry: dict[str, Any], 
    min_level: int = 1, 
    max_level: int = 9
) -> dict[str, list[int]]:
    """
    Compute HEALPix pixel IDs for a GeoJSON geometry at multiple levels.
    
    Optimized to use NumPy arrays throughout.
    
    Args:
        geometry: GeoJSON geometry dictionary (e.g., Point, Polygon, etc.)
        min_level: Minimum HEALPix level (default: 1)
        max_level: Maximum HEALPix level (default: 9)
    
    Returns:
        Dictionary mapping level names to lists of pixel IDs, e.g.,
        {"level_1": [123, 124], "level_2": [456, 457, 458], ...}
    
    Raises:
        ValueError: If geometry is invalid or levels are out of range
    """
    if min_level < 0 or max_level > 29:
        raise ValueError("HEALPix levels must be between 0 and 29")
    
    if min_level > max_level:
        raise ValueError("min_level must be <= max_level")
    
    ids: dict[str, list[int]] = {}
    for lvl in range(min_level, max_level + 1):
        cell_ids = get_healpix_cells_for_geometry(geometry, lvl)
        ids[f"level_{lvl}"] = cell_ids.tolist()  # Convert NumPy array to list for JSON
    
    return ids


def process_single_item(args: tuple[str, dict[str, Any], int, int]) -> tuple[str, dict[str, list[int]]]:
    """
    Process a single item (used by worker processes).
    
    Args:
        args: Tuple of (item_id, geometry, min_level, max_level)
    
    Returns:
        Tuple of (item_id, healpix_data)
    """
    item_id, geometry, min_level, max_level = args
    try:
        healpix_data = compute_healpix_ids(geometry, min_level, max_level)
        return (item_id, healpix_data)
    except Exception as e:
        logger.error(f"Error processing item {item_id}: {e}")
        return (item_id, None)


def process_batch(
    hits: list[dict[str, Any]], 
    index: str,
    min_level: int,
    max_level: int,
    executor: ProcessPoolExecutor = None
) -> tuple[str, int, int, int]:
    """
    Process a batch of Elasticsearch hits and prepare bulk update payload.
    
    Optimized to:
    - Only compute missing levels for items that have partial HEALPix data
    - Build NDJSON more efficiently using list join
    - Return stats about skipped items
    
    Args:
        hits: List of Elasticsearch document hits with geometry data
        index: Name of the Elasticsearch index
        min_level: Minimum HEALPix level to compute
        max_level: Maximum HEALPix level to compute
        executor: Optional ProcessPoolExecutor for parallel processing
    
    Returns:
        Tuple of (NDJSON payload, successful_count, total_cells_computed, skipped_count)
    """
    start_time = time.time()
    bulk_lines: list[str] = []
    successful = 0
    total_cells = 0
    skipped = 0
    
    # Prepare work items - check existing HEALPix data
    work_items = []
    for hit in hits:
        item_id = hit["_id"]
        geometry = hit["_source"]["geometry"]
        existing_healpix = hit["_source"].get("properties", {}).get("dggs:healpix_nested", {})
        
        # Determine which levels need to be computed
        existing_levels = set()
        if existing_healpix:
            for key in existing_healpix.keys():
                if key.startswith("level_"):
                    try:
                        level_num = int(key.split("_")[1])
                        existing_levels.add(level_num)
                    except (IndexError, ValueError):
                        pass
        
        # Check if we need to compute any levels
        needed_levels = set(range(min_level, max_level + 1)) - existing_levels
        
        if not needed_levels:
            skipped += 1
            continue
        
        # Compute only missing levels
        actual_min = min(needed_levels)
        actual_max = max(needed_levels)
        work_items.append((item_id, geometry, actual_min, actual_max, existing_healpix))
    
    if not work_items:
        elapsed = time.time() - start_time
        logger.debug(f"Batch: All {len(hits)} items already have required levels, skipped in {elapsed:.2f}s")
        return "", 0, 0, skipped
    
    # Process in parallel if executor provided
    if executor:
        # Prepare args without existing_healpix for simpler function signature
        compute_items = [(item_id, geom, min_lvl, max_lvl) for item_id, geom, min_lvl, max_lvl, _ in work_items]
        results = list(executor.map(process_single_item, compute_items))
    else:
        compute_items = [(item_id, geom, min_lvl, max_lvl) for item_id, geom, min_lvl, max_lvl, _ in work_items]
        results = [process_single_item(item) for item in compute_items]
    
    # Build bulk payload - merge with existing data
    for (item_id, _, _, _, existing_healpix), (_, new_healpix_data) in zip(work_items, results):
        if new_healpix_data is None:
            continue
        
        # Merge new levels with existing levels and sort by level number
        merged_healpix = {**existing_healpix, **new_healpix_data}
        # Sort by level number to ensure consistent ordering (level_1, level_2, ..., level_11)
        merged_healpix = dict(sorted(merged_healpix.items(), key=lambda x: int(x[0].split('_')[1]) if x[0].startswith('level_') else 0))
        
        # Action line
        bulk_lines.append(json.dumps({
            "update": {"_id": item_id, "_index": index}
        }))
        
        # Document line
        bulk_lines.append(json.dumps({
            "doc": {
                "properties": {
                    "dggs:healpix_nested": merged_healpix
                }
            }
        }))
        
        successful += 1
        
        # Count only new cells
        for cells in new_healpix_data.values():
            total_cells += len(cells)
    
    elapsed = time.time() - start_time
    logger.debug(f"Batch: {len(hits)} items, {successful} updated, {skipped} skipped in {elapsed:.2f}s ({len(hits)/elapsed:.1f} items/s)")
    
    return "\n".join(bulk_lines) + "\n" if bulk_lines else "", successful, total_cells, skipped


async def update_collection_aggregations(
    stac_api_url: str,
    collection_id: str,
    min_level: int,
    max_level: int
) -> None:
    """
    Update collection metadata to include HEALPix aggregation definitions.
    
    Args:
        stac_api_url: STAC API base URL (e.g., http://localhost:8082)
        collection_id: Collection ID to update
        min_level: Minimum HEALPix level
        max_level: Maximum HEALPix level
    
    Raises:
        Exception: If collection fetch or update fails
    """
    logger.info(f"Updating collection '{collection_id}' metadata...")
    
    async with aiohttp.ClientSession() as session:
        # Fetch collection
        get_url = f"{stac_api_url}/collections/{collection_id}"
        async with session.get(get_url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Failed to fetch collection: {resp.status} - {text}")
            collection = await resp.json()
        
        # Default aggregations that should always be present
        default_aggregations = [
            {"name": "total_count", "data_type": "integer"},
            {"name": "datetime_max", "data_type": "datetime"},
            {"name": "datetime_min", "data_type": "datetime"},
            {
                "name": "datetime_frequency",
                "data_type": "frequency_distribution",
                "frequency_distribution_data_type": "datetime",
            },
            {
                "name": "collection_frequency",
                "data_type": "frequency_distribution",
                "frequency_distribution_data_type": "string",
            },
            {
                "name": "geometry_geohash_grid_frequency",
                "data_type": "frequency_distribution",
                "frequency_distribution_data_type": "string",
            },
            {
                "name": "geometry_geotile_grid_frequency",
                "data_type": "frequency_distribution",
                "frequency_distribution_data_type": "string",
            },
            {
                "name": "centroid_geohex_grid_frequency",
                "data_type": "frequency_distribution",
                "frequency_distribution_data_type": "string",
            },
        ]
        
        # Create HEALPix aggregation definitions
        healpix_aggregations = []
        for level in range(min_level, max_level + 1):
            healpix_aggregations.append({
                "name": f"healpix_level_{level}_frequency",
                "data_type": "frequency_distribution",
                "frequency_distribution_data_type": "number"
            })
        
        # Get existing aggregations or initialize empty list
        existing_aggregations = collection.get("aggregations", [])
        
        # Build a set of existing aggregation names
        existing_names = {agg.get("name") for agg in existing_aggregations}
        
        # Add missing default aggregations
        aggregations_to_add = []
        for default_agg in default_aggregations:
            if default_agg["name"] not in existing_names:
                aggregations_to_add.append(default_agg)
        
        # Filter out any existing HEALPix aggregations to avoid duplicates
        non_healpix_aggregations = [
            agg for agg in existing_aggregations 
            if not agg.get("name", "").startswith("healpix_level_")
        ]
        
        # Combine: existing (non-HEALPix) + missing defaults + HEALPix
        collection["aggregations"] = non_healpix_aggregations + aggregations_to_add + healpix_aggregations
        
        if aggregations_to_add:
            logger.info(f"Adding {len(aggregations_to_add)} missing default aggregations...")
        logger.info(f"Adding {len(healpix_aggregations)} HEALPix aggregation definitions...")
        
        # Update collection
        put_url = f"{stac_api_url}/collections/{collection_id}"
        async with session.put(put_url, json=collection) as resp:
            if resp.status not in (200, 204):
                text = await resp.text()
                raise Exception(f"Failed to update collection: {resp.status} - {text}")
        
        logger.info(f"✓ Collection metadata updated with HEALPix aggregations (levels {min_level}-{max_level})")


async def process_all_items(
    es_url: str,
    index: str,
    scroll_size: int,
    scroll_timeout: str,
    min_level: int,
    max_level: int,
    max_workers: int
) -> tuple[int, int]:
    """
    Process all items in the Elasticsearch/OpenSearch index using scroll API.
    
    Optimized to only compute missing levels - if an item has levels 1-10 but you're
    computing 1-11, it will only compute level 11.
    
    Args:
        es_url: Elasticsearch/OpenSearch endpoint URL
        index: Name of the index containing STAC items
        scroll_size: Number of documents per scroll batch
        scroll_timeout: Scroll context timeout (e.g., "5m")
        min_level: Minimum HEALPix level
        max_level: Maximum HEALPix level
        max_workers: Maximum number of concurrent workers
    
    Returns:
        Tuple of (total_processed, total_updated)
    
    Raises:
        aiohttp.ClientError: If HTTP requests fail
    """
    total_processed = 0
    total_updated = 0
    total_skipped = 0
    total_cells = 0
    start_time = time.time()
    
    # Create process pool for parallel HEALPix computation
    executor = ProcessPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    
    try:
        async with aiohttp.ClientSession() as session:
            # Initialize scroll - fetch geometry AND existing HEALPix data
            search_url = f"{es_url}/{index}/_search?scroll={scroll_timeout}"
            search_body = {
                "size": scroll_size,
                "_source": ["geometry", "properties.dggs:healpix_nested"],
                "query": {"match_all": {}}
            }
            
            async with session.post(search_url, json=search_body) as resp:
                resp.raise_for_status()
                data = await resp.json()
                scroll_id: str | None = data.get("_scroll_id")
            
            if not scroll_id:
                raise ValueError("Failed to obtain scroll_id from Elasticsearch/OpenSearch")
            
            total_hits = data["hits"]["total"]["value"]
            logger.info(f"Found {total_hits:,} items to process")
            
            try:
                batch_num = 0
                while True:
                    hits = data["hits"]["hits"]
                    if not hits:
                        break
                    
                    batch_num += 1
                    batch_start = time.time()
                    
                    # Process batch with parallel workers
                    logger.info(f"Batch {batch_num}: Processing {len(hits)} items...")
                    bulk_payload, successful, batch_cells, skipped = process_batch(
                        hits, index, min_level, max_level, executor
                    )
                    total_processed += len(hits)
                    total_cells += batch_cells
                    total_skipped += skipped
                    
                    # Send bulk update if there's anything to update
                    if bulk_payload:
                        bulk_url = f"{es_url}/_bulk"
                        headers = {"Content-Type": "application/x-ndjson"}
                        async with session.post(bulk_url, data=bulk_payload, headers=headers) as resp:
                            resp.raise_for_status()
                            bulk_result = await resp.json()
                            
                            # Count successful updates
                            if not bulk_result.get("errors", False):
                                total_updated += successful
                            else:
                                # Count individual successes
                                for item in bulk_result.get("items", []):
                                    if "update" in item and item["update"].get("status") == 200:
                                        total_updated += 1
                    
                    batch_time = time.time() - batch_start
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_hits - total_processed) / rate if rate > 0 else 0
                    eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                    
                    logger.info(
                        f"Batch {batch_num}: {successful} updated, {skipped} skipped in {batch_time:.1f}s "
                        f"| Total: {total_processed:,}/{total_hits:,} ({total_processed/total_hits*100:.1f}%) "
                        f"| Rate: {rate:.1f} items/s | ETA: {eta_str} "
                        f"| New cells: {batch_cells:,} (avg {batch_cells/successful if successful else 0:.0f}/item)"
                    )
                    
                    # Get next batch
                    scroll_url = f"{es_url}/_search/scroll"
                    scroll_body = {
                        "scroll": scroll_timeout,
                        "scroll_id": scroll_id
                    }
                    async with session.post(scroll_url, json=scroll_body) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        scroll_id = data.get("_scroll_id")
        
            finally:
                # Clean up scroll context
                if scroll_id:
                    delete_scroll_url = f"{es_url}/_search/scroll"
                    try:
                        async with session.delete(delete_scroll_url, json={"scroll_id": scroll_id}):
                            pass
                    except Exception as e:
                        logger.warning(f"Failed to clean up scroll context: {e}")
    finally:
        # Shutdown executor
        if executor:
            executor.shutdown(wait=True)
            logger.info("Worker pool shut down")
    
    logger.info(f"Summary: {total_updated} updated, {total_skipped} skipped (already had required levels)")
    return total_processed, total_updated


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Compute and index HEALPix IDs for STAC items in Elasticsearch/OpenSearch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--es-url",
        required=True,
        help="Elasticsearch/OpenSearch endpoint URL (e.g., https://localhost:9200)"
    )
    
    parser.add_argument(
        "--index",
        required=True,
        help="Name of the Elasticsearch/OpenSearch index containing STAC items"
    )
    
    parser.add_argument(
        "--scroll-size",
        type=int,
        default=10000,
        help="Number of documents to fetch per scroll request"
    )
    
    parser.add_argument(
        "--scroll-timeout",
        default="5m",
        help="Scroll context timeout (e.g., 5m, 1h)"
    )
    
    parser.add_argument(
        "--min-level",
        type=int,
        default=1,
        help="Minimum HEALPix level (0-29)"
    )
    
    parser.add_argument(
        "--max-level",
        type=int,
        default=9,
        help="Maximum HEALPix level (0-29). Note: levels >6 can generate many cells per item, increasing index size significantly."
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for HEALPix computation (default: 4, use 1 to disable)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )
    
    parser.add_argument(
        "--update-collection-metadata",
        action="store_true",
        help="Update collection metadata to add HEALPix aggregation definitions"
    )
    
    parser.add_argument(
        "--stac-api-url",
        help="STAC API base URL for collection metadata update (e.g., http://localhost:8082). Required if --update-collection-metadata is set."
    )
    
    parser.add_argument(
        "--collection-id",
        help="Collection ID for metadata update. If not provided, will be extracted from index name."
    )
    
    return parser.parse_args()


async def async_main() -> int:
    """
    Async main function.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate collection metadata update arguments
    if args.update_collection_metadata:
        if not args.stac_api_url:
            logger.error("--stac-api-url is required when --update-collection-metadata is set")
            return 1
        
        # Extract collection ID from index name if not provided
        if not args.collection_id:
            # Try to extract from index name (e.g., items_sentinel-2-l2a_... -> sentinel-2-l2a)
            if args.index.startswith("items_"):
                parts = args.index.split("_", 2)
                if len(parts) >= 2:
                    args.collection_id = parts[1]
                else:
                    logger.error("Could not extract collection ID from index name. Please provide --collection-id")
                    return 1
            else:
                logger.error("--collection-id is required when index name doesn't follow items_<collection>_* pattern")
                return 1
        
        logger.info(f"Collection metadata will be updated for: {args.collection_id}")
    
    # Warn about high levels
    if args.max_level > 6:
        print(f"⚠️  Warning: Using max_level={args.max_level} may generate many cells per item.")
        print(f"   This can significantly increase index size and query costs.")
        print(f"   Consider using --max-level 6 unless you need fine-grained aggregations.")
        print()
    
    try:
        logger.info("="*60)
        logger.info("Starting HEALPix computation")
        logger.info(f"Index: {args.index}")
        logger.info(f"Elasticsearch/OpenSearch URL: {args.es_url}")
        logger.info(f"HEALPix levels: {args.min_level} to {args.max_level}")
        logger.info(f"Scroll size: {args.scroll_size}")
        logger.info(f"Parallel workers: {args.max_workers}")
        logger.info(f"Method: Full geometry intersection (not centroids)")
        logger.info("="*60)
        
        # Update collection metadata if requested (do this before processing items)
        if args.update_collection_metadata:
            try:
                await update_collection_aggregations(
                    stac_api_url=args.stac_api_url,
                    collection_id=args.collection_id,
                    min_level=args.min_level,
                    max_level=args.max_level
                )
            except Exception as e:
                logger.error(f"Failed to update collection metadata: {e}")
                logger.warning("Continuing with item processing...")
        
        total_processed, total_updated = await process_all_items(
            es_url=args.es_url,
            index=args.index,
            scroll_size=args.scroll_size,
            scroll_timeout=args.scroll_timeout,
            min_level=args.min_level,
            max_level=args.max_level,
            max_workers=args.max_workers
        )
        
        logger.info("="*60)
        logger.info("✓ Completed successfully!")
        logger.info(f"Total processed: {total_processed:,}")
        logger.info(f"Total updated: {total_updated:,}")
        if total_processed > 0:
            elapsed = time.time() - __import__('time').time()
            logger.info(f"Success rate: {total_updated/total_processed*100:.1f}%")
        logger.info("="*60)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        logger.info(f"Processed {total_processed:,} items before interruption")
        return 130
    
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=args.verbose)
        return 1


def main() -> None:
    """Main entry point for the CLI."""
    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
