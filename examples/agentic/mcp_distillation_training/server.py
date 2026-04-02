"""ShopInsights Analytics Platform — FastMCP server with 15 e-commerce tools.

Start the server:
    uv run python examples/agentic/ecommerce_mcp/server.py

The server exposes a Streamable-HTTP transport on port 8000 (configurable via
the ``PORT`` environment variable) for Langflow / other MCP clients.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Annotated, Any, Optional
import os
import statistics
import uuid

from data import DataStore, create_data_store
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="ShopInsights Analytics Platform",
    instructions=(
        "E-commerce analytics platform for an online retailer. "
        "Provides product search, sales analytics, customer insights, "
        "demand forecasting, and promotional management."
    ),
)

store: DataStore = create_data_store()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _in_range(date_str: str, start: str, end: str) -> bool:
    d = _parse_date(date_str)
    return _parse_date(start) <= d <= _parse_date(end)


def _product_by_id(pid: str) -> dict[str, Any] | None:
    for p in store.products:
        if p["id"] == pid:
            return p
    return None


def _match_category(product_cat: str, query_cat: str) -> bool:
    """Check if product category matches (exact or parent match)."""
    return (
        product_cat == query_cat
        or product_cat.startswith(query_cat + " >")
        or product_cat.endswith("> " + query_cat)
    )


# =========================================================================
# Cluster 1 — Product Discovery
# =========================================================================
@mcp.tool()
def search_products(
    query: Annotated[
        str, "Full-text search query for product name, description, or tags"
    ],
    category: Annotated[
        Optional[str], "Exact category name to filter (e.g. 'Electronics > Phones')"
    ] = None,
    min_price: Annotated[Optional[float], "Minimum price filter"] = None,
    max_price: Annotated[Optional[float], "Maximum price filter"] = None,
    brand: Annotated[Optional[str], "Filter by brand name"] = None,
    in_stock_only: Annotated[bool, "Only return products currently in stock"] = True,
    sort_by: Annotated[
        str,
        "Sort order: relevance, price_asc, price_desc, rating, newest",
    ] = "relevance",
    limit: Annotated[int, "Max results to return (1-50)"] = 10,
) -> dict[str, Any]:
    """Search the product catalog using keywords and filters.

    Returns a paginated list of products matching the query. Use this when
    a user is looking for products by name, keyword, or specific criteria.
    """
    q = query.lower()
    results = []
    for p in store.products:
        text = f"{p['name']} {' '.join(p['tags'])} {p['brand']}".lower()
        if q not in text:
            continue
        if category and not _match_category(p["category"], category):
            continue
        if min_price is not None and p["price"] < min_price:
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        if brand and p["brand"].lower() != brand.lower():
            continue
        if in_stock_only:
            total_stock = sum(
                inv["quantity"]
                for inv in store.inventory
                if inv["product_id"] == p["id"]
            )
            if total_stock <= 0:
                continue
        results.append(p)

    sort_keys = {
        "price_asc": lambda x: x["price"],
        "price_desc": lambda x: -x["price"],
        "rating": lambda x: -x["avg_rating"],
        "newest": lambda x: x["created_at"],
    }
    if sort_by in sort_keys:
        results.sort(key=sort_keys[sort_by], reverse=(sort_by == "newest"))

    limit = max(1, min(limit, 50))
    return {
        "total": len(results),
        "products": results[:limit],
        "query": query,
        "filters_applied": {
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
            "brand": brand,
            "in_stock_only": in_stock_only,
        },
    }


@mcp.tool()
def browse_catalog(
    category: Annotated[
        str, "Category to browse (e.g. 'Electronics', 'Electronics > Phones')"
    ],
    sort_by: Annotated[
        str, "Sort: price_asc, price_desc, rating, newest, popularity"
    ] = "popularity",
    limit: Annotated[int, "Max results (1-50)"] = 20,
) -> dict[str, Any]:
    """Browse products by category hierarchy.

    Use this for exploratory browsing when the user wants to see what's
    available in a category, rather than searching for something specific.
    """
    results = [p for p in store.products if _match_category(p["category"], category)]

    sort_keys: dict[str, Any] = {
        "price_asc": lambda x: x["price"],
        "price_desc": lambda x: -x["price"],
        "rating": lambda x: -x["avg_rating"],
        "newest": lambda x: x["created_at"],
        "popularity": lambda x: -x["review_count"],
    }
    if sort_by in sort_keys:
        results.sort(key=sort_keys[sort_by], reverse=(sort_by == "newest"))

    subcategories = sorted(
        {
            p["category"].split(" > ")[-1]
            for p in store.products
            if _match_category(p["category"], category) and p["category"] != category
        }
    )

    limit = max(1, min(limit, 50))
    return {
        "category": category,
        "total": len(results),
        "subcategories": subcategories,
        "products": results[:limit],
    }


@mcp.tool()
def get_trending_products(
    metric: Annotated[
        str,
        "Trending metric: sales, views, conversion, revenue",
    ] = "sales",
    category: Annotated[Optional[str], "Limit to a specific category"] = None,
    days: Annotated[int, "Lookback window in days (1-90)"] = 30,
    limit: Annotated[int, "Max results (1-20)"] = 10,
) -> dict[str, Any]:
    """Get products with rising performance metrics in a recent time window.

    Use this when the user wants to know what's hot, trending, or gaining
    traction — not for general search.
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Aggregate per-product sales from orders in the window
    product_metrics: dict[str, dict[str, float]] = defaultdict(
        lambda: {"sales": 0, "revenue": 0.0, "orders": 0}
    )
    for order in store.orders:
        if order["date"] < cutoff:
            continue
        for item in order["items"]:
            pid = item["product_id"]
            product_metrics[pid]["sales"] += item["quantity"]
            product_metrics[pid]["revenue"] += item["quantity"] * item["unit_price"]
            product_metrics[pid]["orders"] += 1

    scored: list[tuple[dict[str, Any], float]] = []
    for p in store.products:
        if category and not _match_category(p["category"], category):
            continue
        m = product_metrics.get(p["id"], {"sales": 0, "revenue": 0.0, "orders": 0})
        if metric == "sales":
            score = m["sales"]
        elif metric == "revenue":
            score = m["revenue"]
        elif metric == "conversion":
            score = m["orders"] / max(p["review_count"], 1) * 100
        else:  # views proxy
            score = float(p["review_count"])
        scored.append((p, score))

    scored.sort(key=lambda x: -x[1])
    limit = max(1, min(limit, 20))
    return {
        "metric": metric,
        "period_days": days,
        "trending": [
            {**prod, "trend_score": round(sc, 2)} for prod, sc in scored[:limit]
        ],
    }


@mcp.tool()
def get_product_details(
    product_id: Annotated[str, "Product ID (e.g. 'PROD-0001')"],
) -> dict[str, Any]:
    """Get full details for a single product by its ID.

    Returns comprehensive information including price, cost, specifications,
    images placeholder, and review summary. Requires a product ID obtained
    from search, browse, or trending results.
    """
    product = _product_by_id(product_id)
    if not product:
        return {"error": f"Product '{product_id}' not found"}

    # Compute sales stats
    total_sold = 0
    total_revenue = 0.0
    for order in store.orders:
        for item in order["items"]:
            if item["product_id"] == product_id:
                total_sold += item["quantity"]
                total_revenue += item["quantity"] * item["unit_price"]

    total_stock = sum(
        inv["quantity"] for inv in store.inventory if inv["product_id"] == product_id
    )

    return {
        **product,
        "total_units_sold": total_sold,
        "total_revenue": round(total_revenue, 2),
        "total_stock": total_stock,
        "margin_pct": round(
            (product["price"] - product["cost"]) / product["price"] * 100, 1
        ),
    }


# =========================================================================
# Cluster 2 — Sales & Revenue
# =========================================================================
@mcp.tool()
def get_sales_data(
    product_ids: Annotated[list[str], "List of product IDs (1-10)"],
    date_from: Annotated[str, "Start date (ISO format YYYY-MM-DD)"],
    date_to: Annotated[str, "End date (ISO format YYYY-MM-DD)"],
    granularity: Annotated[str, "Aggregation: daily, weekly, monthly"] = "daily",
) -> dict[str, Any]:
    """Get unit sales data for specific products over a date range.

    Returns time-series sales data at the requested granularity. Use this
    for product-level, unit-focused analysis.
    """
    if len(product_ids) > 10:
        return {"error": "Maximum 10 product IDs allowed"}

    # Aggregate by date bucket
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for order in store.orders:
        if not _in_range(order["date"], date_from, date_to):
            continue
        dt = _parse_date(order["date"])
        if granularity == "weekly":
            key = (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")
        elif granularity == "monthly":
            key = dt.strftime("%Y-%m")
        else:
            key = order["date"]
        for item in order["items"]:
            if item["product_id"] in product_ids:
                buckets[key][item["product_id"]] += item["quantity"]

    time_series = [
        {"period": k, "sales_by_product": dict(v)} for k, v in sorted(buckets.items())
    ]

    return {
        "product_ids": product_ids,
        "date_from": date_from,
        "date_to": date_to,
        "granularity": granularity,
        "time_series": time_series,
        "total_units": sum(sum(v.values()) for v in buckets.values()),
    }


@mcp.tool()
def get_revenue_report(
    date_from: Annotated[str, "Start date (ISO format YYYY-MM-DD)"],
    date_to: Annotated[str, "End date (ISO format YYYY-MM-DD)"],
    group_by: Annotated[
        str,
        "Dimension to group by: category, region, channel",
    ] = "category",
) -> dict[str, Any]:
    """Get aggregated revenue breakdown by a business dimension.

    Returns total revenue grouped by category, region, or channel. Use this
    for high-level business reporting, not for per-product analysis.
    """
    breakdown: dict[str, dict[str, float]] = defaultdict(
        lambda: {"revenue": 0.0, "orders": 0, "units": 0}
    )

    for order in store.orders:
        if not _in_range(order["date"], date_from, date_to):
            continue
        if group_by == "region":
            key = order["region"]
        elif group_by == "channel":
            key = order["channel"]
        else:
            # Group by top-level category
            for item in order["items"]:
                prod = _product_by_id(item["product_id"])
                cat = prod["category"].split(" > ")[0] if prod else "Unknown"
                breakdown[cat]["revenue"] += item["unit_price"] * item["quantity"]
                breakdown[cat]["units"] += item["quantity"]
            breakdown[order.get("region", "Unknown")]["orders"] += 0  # counted below
            # Count order once
            continue

        breakdown[key]["revenue"] += order["total"]
        breakdown[key]["orders"] += 1
        breakdown[key]["units"] += sum(i["quantity"] for i in order["items"])

    # For category grouping, count orders separately
    if group_by == "category":
        for order in store.orders:
            if not _in_range(order["date"], date_from, date_to):
                continue
            cats_in_order = set()
            for item in order["items"]:
                prod = _product_by_id(item["product_id"])
                if prod:
                    cats_in_order.add(prod["category"].split(" > ")[0])
            for cat in cats_in_order:
                breakdown[cat]["orders"] += 1

    result = {
        "date_from": date_from,
        "date_to": date_to,
        "group_by": group_by,
        "breakdown": {
            k: {
                kk: round(vv, 2) if isinstance(vv, float) else vv
                for kk, vv in v.items()
            }
            for k, v in sorted(breakdown.items())
        },
        "total_revenue": round(sum(v["revenue"] for v in breakdown.values()), 2),
    }
    return result


@mcp.tool()
def get_store_overview() -> dict[str, Any]:
    """Get a quick dashboard snapshot of overall store performance.

    Returns total revenue, order count, average order value, conversion rate,
    and top-5 products. Takes no parameters — use this for a fast summary.
    """
    total_revenue = sum(o["total"] for o in store.orders)
    total_orders = len(store.orders)
    completed = [o for o in store.orders if o["status"] == "completed"]
    aov = total_revenue / total_orders if total_orders else 0

    # Top 5 products by revenue
    prod_rev: dict[str, float] = defaultdict(float)
    for order in store.orders:
        for item in order["items"]:
            prod_rev[item["product_id"]] += item["unit_price"] * item["quantity"]

    top5 = sorted(prod_rev.items(), key=lambda x: -x[1])[:5]
    top5_details = []
    for pid, rev in top5:
        p = _product_by_id(pid)
        top5_details.append(
            {
                "product_id": pid,
                "name": p["name"] if p else "Unknown",
                "revenue": round(rev, 2),
            }
        )

    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "completed_orders": len(completed),
        "average_order_value": round(aov, 2),
        "conversion_rate_pct": round(len(completed) / total_orders * 100, 1)
        if total_orders
        else 0,
        "top_5_products": top5_details,
        "total_products": len(store.products),
        "total_customers": len(store.customers),
    }


# =========================================================================
# Cluster 3 — Customer Analytics
# =========================================================================
@mcp.tool()
def get_customer_segments(
    segment_by: Annotated[
        str,
        "Segmentation dimension: spending_tier, recency, frequency",
    ] = "spending_tier",
) -> dict[str, Any]:
    """Get aggregate customer segment breakdown.

    Returns segment sizes, average lifetime values, and order counts.
    Use this for understanding customer populations, not individual customers.
    """
    segments: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for c in store.customers:
        if segment_by == "spending_tier":
            ltv = c["lifetime_value"]
            if ltv > 3000:
                tier = "high"
            elif ltv > 1000:
                tier = "medium"
            else:
                tier = "low"
        elif segment_by == "recency":
            tier = c["segment"]  # reuse stored segment as proxy
        else:
            orders = c["total_orders"]
            if orders > 20:
                tier = "frequent"
            elif orders > 5:
                tier = "regular"
            else:
                tier = "occasional"
        segments[tier].append(c)

    result = {}
    for tier, members in sorted(segments.items()):
        ltvs = [m["lifetime_value"] for m in members]
        result[tier] = {
            "count": len(members),
            "avg_lifetime_value": round(statistics.mean(ltvs), 2) if ltvs else 0,
            "total_lifetime_value": round(sum(ltvs), 2),
            "avg_orders": round(
                statistics.mean([m["total_orders"] for m in members]), 1
            ),
        }

    return {"segment_by": segment_by, "segments": result}


@mcp.tool()
def get_customer_profile(
    customer_id: Annotated[Optional[str], "Customer ID (e.g. 'CUST-0001')"] = None,
    email: Annotated[Optional[str], "Customer email address"] = None,
) -> dict[str, Any]:
    """Get a single customer's full profile.

    Includes order history, lifetime value, and segment. Provide either
    customer_id or email (at least one required).
    """
    customer = None
    for c in store.customers:
        if (customer_id and c["id"] == customer_id) or (email and c["email"] == email):
            customer = c
            break
    if not customer:
        return {"error": "Customer not found"}

    orders = [o for o in store.orders if o["customer_id"] == customer["id"]]
    recent_orders = sorted(orders, key=lambda x: x["date"], reverse=True)[:5]

    return {
        **customer,
        "order_count": len(orders),
        "total_spent": round(sum(o["total"] for o in orders), 2),
        "recent_orders": recent_orders,
    }


@mcp.tool()
def get_abandoned_carts(
    days: Annotated[int, "Lookback window in days"] = 30,
    min_value: Annotated[Optional[float], "Minimum cart value filter"] = None,
) -> dict[str, Any]:
    """Get abandoned shopping carts within a time window.

    Returns carts that were created but never converted to orders.
    Use this to identify recovery opportunities.
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    results = []
    for cart in store.abandoned_carts:
        if cart["created_at"] < cutoff:
            continue
        if min_value is not None and cart["total_value"] < min_value:
            continue
        results.append(cart)

    results.sort(key=lambda x: -x["total_value"])
    return {
        "period_days": days,
        "total_carts": len(results),
        "total_value": round(sum(c["total_value"] for c in results), 2),
        "carts": results,
    }


# =========================================================================
# Multi-step & Complex Tools
# =========================================================================
@mcp.tool()
def analyze_product_performance(
    product_id: Annotated[str, "Product ID to analyze"],
) -> dict[str, Any]:
    """Analyze detailed performance metrics for a single product.

    Returns conversion rate, return rate, average rating, category rank,
    and sell-through rate. Requires a product_id (obtain from search or
    trending tools first).
    """
    product = _product_by_id(product_id)
    if not product:
        return {"error": f"Product '{product_id}' not found"}

    # Sales & returns
    total_sold = 0
    total_returned = 0
    total_orders_with_product = 0
    for order in store.orders:
        for item in order["items"]:
            if item["product_id"] == product_id:
                total_sold += item["quantity"]
                total_orders_with_product += 1
                if order["status"] == "returned":
                    total_returned += item["quantity"]

    # Category peers
    cat = product["category"]
    peers = [p for p in store.products if p["category"] == cat]
    peers_sorted = sorted(peers, key=lambda x: -x["avg_rating"])
    rank = next(
        (i + 1 for i, p in enumerate(peers_sorted) if p["id"] == product_id), len(peers)
    )

    # Inventory
    total_stock = sum(
        inv["quantity"] for inv in store.inventory if inv["product_id"] == product_id
    )
    total_supply = total_stock + total_sold
    sell_through = round(total_sold / total_supply * 100, 1) if total_supply > 0 else 0

    return {
        "product_id": product_id,
        "product_name": product["name"],
        "category": cat,
        "total_units_sold": total_sold,
        "total_orders": total_orders_with_product,
        "return_rate_pct": round(total_returned / total_sold * 100, 1)
        if total_sold > 0
        else 0,
        "avg_rating": product["avg_rating"],
        "review_count": product["review_count"],
        "category_rank": f"{rank}/{len(peers)}",
        "sell_through_rate_pct": sell_through,
        "current_stock": total_stock,
        "margin_pct": round(
            (product["price"] - product["cost"]) / product["price"] * 100, 1
        ),
    }


@mcp.tool()
def compare_products(
    product_ids: Annotated[list[str], "List of 2-5 product IDs to compare"],
) -> dict[str, Any]:
    """Compare multiple products side-by-side across all metrics.

    Requires 2-5 product IDs. Typically chained after a search or browse
    to compare shortlisted items.
    """
    if len(product_ids) < 2:
        return {"error": "Need at least 2 product IDs"}
    if len(product_ids) > 5:
        return {"error": "Maximum 5 product IDs for comparison"}

    comparisons = []
    for pid in product_ids:
        product = _product_by_id(pid)
        if not product:
            comparisons.append({"product_id": pid, "error": "Not found"})
            continue

        total_sold = sum(
            item["quantity"]
            for order in store.orders
            for item in order["items"]
            if item["product_id"] == pid
        )
        total_revenue = sum(
            item["quantity"] * item["unit_price"]
            for order in store.orders
            for item in order["items"]
            if item["product_id"] == pid
        )
        total_stock = sum(
            inv["quantity"] for inv in store.inventory if inv["product_id"] == pid
        )

        comparisons.append(
            {
                "product_id": pid,
                "name": product["name"],
                "category": product["category"],
                "brand": product["brand"],
                "price": product["price"],
                "cost": product["cost"],
                "margin_pct": round(
                    (product["price"] - product["cost"]) / product["price"] * 100, 1
                ),
                "avg_rating": product["avg_rating"],
                "review_count": product["review_count"],
                "total_units_sold": total_sold,
                "total_revenue": round(total_revenue, 2),
                "current_stock": total_stock,
            }
        )

    return {"comparison": comparisons, "product_count": len(comparisons)}


@mcp.tool()
def forecast_demand(
    product_id: Annotated[str, "Product ID to forecast"],
    days: Annotated[int, "Number of days to forecast (1-90)"] = 30,
) -> dict[str, Any]:
    """Predict demand for a product over the next N days.

    Returns daily predictions with confidence intervals based on
    historical sales patterns. Requires a product_id.
    """
    product = _product_by_id(product_id)
    if not product:
        return {"error": f"Product '{product_id}' not found"}

    days = max(1, min(days, 90))

    # Historical daily sales rate
    daily_sales: dict[str, int] = defaultdict(int)
    for order in store.orders:
        for item in order["items"]:
            if item["product_id"] == product_id:
                daily_sales[order["date"]] += item["quantity"]

    if daily_sales:
        values = list(daily_sales.values())
        avg_daily = statistics.mean(values)
        std_daily = statistics.stdev(values) if len(values) > 1 else avg_daily * 0.3
    else:
        avg_daily = 0.5
        std_daily = 0.3

    import random as _rng

    _rng.seed(hash(product_id) + days)
    predictions = []
    today = datetime.now()
    for i in range(1, days + 1):
        date = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        predicted = max(0, round(avg_daily + _rng.gauss(0, std_daily * 0.3), 1))
        lower = max(0, round(predicted - 1.96 * std_daily, 1))
        upper = round(predicted + 1.96 * std_daily, 1)
        predictions.append(
            {
                "date": date,
                "predicted_units": predicted,
                "confidence_interval": {"lower": lower, "upper": upper},
            }
        )

    total_predicted = round(sum(p["predicted_units"] for p in predictions), 1)
    total_stock = sum(
        inv["quantity"] for inv in store.inventory if inv["product_id"] == product_id
    )

    return {
        "product_id": product_id,
        "product_name": product["name"],
        "forecast_days": days,
        "avg_daily_demand": round(avg_daily, 2),
        "total_predicted_demand": total_predicted,
        "current_stock": total_stock,
        "stock_sufficient": total_stock >= total_predicted,
        "predictions": predictions[:14],  # Only return first 14 days in detail
        "note": f"Showing first 14 of {days} daily predictions" if days > 14 else None,
    }


@mcp.tool()
def get_inventory_status(
    product_ids: Annotated[
        Optional[list[str]], "Product IDs to check (omit for all)"
    ] = None,
    warehouse: Annotated[Optional[str], "Filter by warehouse name"] = None,
    low_stock_only: Annotated[
        bool, "Only show items at or below reorder point"
    ] = False,
) -> dict[str, Any]:
    """Get stock levels per warehouse for products.

    Includes reorder point and days-of-supply estimates. Use this before
    making promotion or restocking decisions.
    """
    results = []
    for inv in store.inventory:
        if product_ids and inv["product_id"] not in product_ids:
            continue
        if warehouse and inv["warehouse"] != warehouse:
            continue
        if low_stock_only and inv["quantity"] > inv["reorder_point"]:
            continue
        results.append(inv)

    return {
        "total_entries": len(results),
        "inventory": results,
        "warehouses": sorted({r["warehouse"] for r in results}),
        "low_stock_count": sum(
            1 for r in results if r["quantity"] <= r["reorder_point"]
        ),
    }


@mcp.tool()
def create_promotion(
    name: Annotated[str, "Promotion name"],
    product_ids: Annotated[list[str], "Product IDs to include in promotion"],
    discount_type: Annotated[str, "Type: percentage, fixed_amount, buy_one_get_one"],
    discount_value: Annotated[
        float, "Discount value (percentage or fixed amount; ignored for BOGO)"
    ] = 0.0,
    start_date: Annotated[str, "Start date (ISO format YYYY-MM-DD)"] = "",
    end_date: Annotated[str, "End date (ISO format YYYY-MM-DD)"] = "",
    min_quantity: Annotated[Optional[int], "Minimum quantity condition"] = None,
    min_order_value: Annotated[Optional[float], "Minimum order value condition"] = None,
    customer_segment: Annotated[
        str, "Target segment: all, vip, new, returning"
    ] = "all",
) -> dict[str, Any]:
    """Create a new promotional discount campaign.

    Complex action that creates a discount for specified products. You should
    check inventory and product performance before creating promotions.
    Validates product IDs and date ranges.
    """
    # Validate products exist
    valid_ids = []
    invalid_ids = []
    for pid in product_ids:
        if _product_by_id(pid):
            valid_ids.append(pid)
        else:
            invalid_ids.append(pid)

    if invalid_ids:
        return {"error": f"Invalid product IDs: {invalid_ids}"}

    if discount_type not in ("percentage", "fixed_amount", "buy_one_get_one"):
        return {"error": f"Invalid discount_type: {discount_type}"}

    if discount_type == "percentage" and not (0 < discount_value <= 100):
        return {"error": "Percentage discount must be between 0 and 100"}

    if not start_date or not end_date:
        return {"error": "start_date and end_date are required"}

    try:
        s = _parse_date(start_date)
        e = _parse_date(end_date)
        if e <= s:
            return {"error": "end_date must be after start_date"}
    except ValueError:
        return {"error": "Invalid date format, use YYYY-MM-DD"}

    conditions: dict[str, Any] = {"customer_segment": customer_segment}
    if min_quantity is not None:
        conditions["min_quantity"] = min_quantity
    if min_order_value is not None:
        conditions["min_order_value"] = min_order_value

    promo = {
        "id": f"PROMO-{uuid.uuid4().hex[:8].upper()}",
        "name": name,
        "product_ids": valid_ids,
        "discount_type": discount_type,
        "discount_value": discount_value,
        "start_date": start_date,
        "end_date": end_date,
        "active": True,
        "conditions": conditions,
    }
    store.promotions.append(promo)

    return {
        "status": "created",
        "promotion": promo,
        "products_count": len(valid_ids),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8008"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
