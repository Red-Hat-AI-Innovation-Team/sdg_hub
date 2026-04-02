"""Deterministic seed data for the ShopInsights Analytics Platform MCP server.

Generates products, customers, orders, inventory, abandoned carts, and promotions
with a fixed random seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import random

SEED = 42

# ---------------------------------------------------------------------------
# Category hierarchy
# ---------------------------------------------------------------------------
CATEGORY_TREE: dict[str, list[str]] = {
    "Electronics": ["Phones", "Laptops", "Accessories"],
    "Home & Kitchen": ["Appliances", "Furniture"],
    "Clothing": ["Men", "Women"],
    "Sports & Outdoors": ["Fitness", "Camping"],
}

FLAT_CATEGORIES: list[str] = [
    f"{parent} > {child}"
    for parent, children in CATEGORY_TREE.items()
    for child in children
]

# ---------------------------------------------------------------------------
# Brand / product name pools (keyed by sub-category)
# ---------------------------------------------------------------------------
BRANDS: dict[str, list[str]] = {
    "Phones": ["TechPulse", "NovaPhone", "ZenMobile"],
    "Laptops": ["ByteForce", "AeroBook", "CompuMax"],
    "Accessories": ["GadgetGo", "SnapGear", "LinkWare"],
    "Appliances": ["HomeZen", "KitchenPro", "EasyCook"],
    "Furniture": ["ComfortPlus", "WoodCraft", "UrbanLoft"],
    "Men": ["StreetStyle", "ClassicFit", "UrbanWear"],
    "Women": ["ChicBloom", "VelvetEdge", "PureThread"],
    "Fitness": ["FlexCore", "IronGrip", "PeakMove"],
    "Camping": ["TrailBoss", "WildPath", "CampEase"],
}

PRODUCT_TEMPLATES: dict[str, list[str]] = {
    "Phones": [
        "{brand} Pro Max 15",
        "{brand} Lite SE",
        "{brand} Ultra 7",
        "{brand} Flip Z",
        "{brand} Edge Plus",
    ],
    "Laptops": [
        "{brand} Ultrabook 14",
        "{brand} Gaming X17",
        "{brand} Workstation Pro",
        "{brand} Chromebook Air",
        "{brand} Studio 16",
    ],
    "Accessories": [
        "{brand} Wireless Earbuds",
        "{brand} USB-C Hub",
        "{brand} Phone Case",
        "{brand} Power Bank 20K",
        "{brand} Screen Protector 2-Pack",
    ],
    "Appliances": [
        "{brand} Air Fryer 5L",
        "{brand} Blender Pro",
        "{brand} Coffee Maker Deluxe",
        "{brand} Toaster Oven",
        "{brand} Electric Kettle",
    ],
    "Furniture": [
        "{brand} Ergonomic Desk Chair",
        '{brand} Standing Desk 60"',
        "{brand} Bookshelf Oak",
        "{brand} TV Console Walnut",
        "{brand} Side Table",
    ],
    "Men": [
        "{brand} Slim Chinos",
        "{brand} Henley Tee",
        "{brand} Bomber Jacket",
        "{brand} Oxford Shirt",
        "{brand} Running Shorts",
    ],
    "Women": [
        "{brand} Midi Wrap Dress",
        "{brand} Yoga Leggings",
        "{brand} Denim Jacket",
        "{brand} Cashmere Sweater",
        "{brand} Linen Blouse",
    ],
    "Fitness": [
        "{brand} Resistance Band Set",
        "{brand} Adjustable Dumbbells",
        "{brand} Yoga Mat Premium",
        "{brand} Jump Rope Speed",
        "{brand} Foam Roller",
    ],
    "Camping": [
        "{brand} 4-Person Tent",
        "{brand} Sleeping Bag -10C",
        "{brand} Portable Stove",
        "{brand} Headlamp 1000lm",
        "{brand} Water Filter",
    ],
}

PRICE_RANGES: dict[str, tuple[float, float]] = {
    "Phones": (199.0, 1299.0),
    "Laptops": (399.0, 2499.0),
    "Accessories": (9.99, 89.99),
    "Appliances": (29.99, 249.99),
    "Furniture": (79.99, 699.99),
    "Men": (19.99, 149.99),
    "Women": (24.99, 199.99),
    "Fitness": (14.99, 199.99),
    "Camping": (24.99, 399.99),
}

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
CHANNELS = ["website", "mobile_app", "marketplace", "social_media"]
WAREHOUSES = ["Warehouse-East", "Warehouse-West", "Warehouse-Central"]
ORDER_STATUSES = ["completed", "processing", "shipped", "returned", "cancelled"]
SEGMENTS = ["vip", "returning", "new", "at_risk"]

TAGS_POOL: dict[str, list[str]] = {
    "Phones": ["5G", "OLED", "waterproof", "fast-charge"],
    "Laptops": ["SSD", "16GB-RAM", "backlit-keyboard", "thunderbolt"],
    "Accessories": ["compact", "wireless", "durable", "travel-friendly"],
    "Appliances": ["energy-efficient", "dishwasher-safe", "stainless-steel"],
    "Furniture": ["assembly-required", "solid-wood", "adjustable", "modern"],
    "Men": ["cotton", "slim-fit", "machine-washable"],
    "Women": ["breathable", "stretchy", "sustainable"],
    "Fitness": ["portable", "non-slip", "adjustable-resistance"],
    "Camping": ["waterproof", "lightweight", "compact-pack"],
}


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------
def _uid(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx:04d}"


def _iso_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _random_date(rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, max(delta, 1)))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def _generate_products(rng: random.Random) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    idx = 1
    for parent, children in CATEGORY_TREE.items():
        for child in children:
            category = f"{parent} > {child}"
            brands = BRANDS[child]
            templates = PRODUCT_TEMPLATES[child]
            lo, hi = PRICE_RANGES[child]
            tags_pool = TAGS_POOL[child]

            combos = [(b, t) for b in brands for t in templates]
            rng.shuffle(combos)
            # Pick ~5-6 per sub-category
            count = min(len(combos), rng.randint(5, 6))
            for brand, tmpl in combos[:count]:
                price = round(rng.uniform(lo, hi), 2)
                cost = round(price * rng.uniform(0.35, 0.65), 2)
                products.append(
                    {
                        "id": _uid("PROD", idx),
                        "name": tmpl.format(brand=brand),
                        "category": category,
                        "brand": brand,
                        "price": price,
                        "cost": cost,
                        "tags": rng.sample(
                            tags_pool, k=min(rng.randint(1, 3), len(tags_pool))
                        ),
                        "specs": {
                            "weight_kg": round(rng.uniform(0.1, 15.0), 2),
                            "color": rng.choice(
                                ["Black", "White", "Silver", "Blue", "Red"]
                            ),
                        },
                        "avg_rating": round(rng.uniform(3.0, 5.0), 1),
                        "review_count": rng.randint(5, 2000),
                        "created_at": _iso_date(
                            _random_date(
                                rng, datetime(2024, 1, 1), datetime(2025, 6, 1)
                            )
                        ),
                    }
                )
                idx += 1
    return products


def _generate_customers(rng: random.Random) -> list[dict[str, Any]]:
    first_names = [
        "Alice",
        "Bob",
        "Carlos",
        "Diana",
        "Ethan",
        "Fiona",
        "George",
        "Hannah",
        "Ivan",
        "Julia",
        "Kevin",
        "Lena",
        "Miguel",
        "Nora",
        "Oliver",
        "Priya",
        "Quinn",
        "Rosa",
        "Sam",
        "Tina",
        "Uma",
        "Victor",
        "Wendy",
        "Xavier",
        "Yuki",
        "Zara",
        "Aaron",
        "Bella",
        "Chris",
        "Dana",
    ]
    customers: list[dict[str, Any]] = []
    for i, name in enumerate(first_names, 1):
        segment = rng.choice(SEGMENTS)
        total_orders = rng.randint(1, 50)
        ltv = round(total_orders * rng.uniform(30.0, 200.0), 2)
        customers.append(
            {
                "id": _uid("CUST", i),
                "email": f"{name.lower()}@example.com",
                "name": name,
                "segment": segment,
                "lifetime_value": ltv,
                "total_orders": total_orders,
                "joined_at": _iso_date(
                    _random_date(rng, datetime(2022, 1, 1), datetime(2025, 3, 1))
                ),
            }
        )
    return customers


def _generate_orders(
    rng: random.Random,
    products: list[dict[str, Any]],
    customers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for i in range(1, 201):
        customer = rng.choice(customers)
        num_items = rng.randint(1, 4)
        items_products = rng.sample(products, k=min(num_items, len(products)))
        items = []
        for p in items_products:
            qty = rng.randint(1, 3)
            items.append(
                {
                    "product_id": p["id"],
                    "product_name": p["name"],
                    "quantity": qty,
                    "unit_price": p["price"],
                }
            )
        order_date = _random_date(rng, datetime(2025, 1, 1), datetime(2025, 12, 31))
        orders.append(
            {
                "id": _uid("ORD", i),
                "customer_id": customer["id"],
                "customer_name": customer["name"],
                "items": items,
                "date": _iso_date(order_date),
                "status": rng.choice(ORDER_STATUSES),
                "channel": rng.choice(CHANNELS),
                "region": rng.choice(REGIONS),
                "total": round(
                    sum(it["unit_price"] * it["quantity"] for it in items), 2
                ),
            }
        )
    return orders


def _generate_inventory(
    rng: random.Random, products: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for p in products:
        for wh in WAREHOUSES:
            qty = rng.randint(0, 500)
            reorder = rng.randint(10, 50)
            daily_sales = rng.uniform(0.5, 10.0)
            inventory.append(
                {
                    "product_id": p["id"],
                    "product_name": p["name"],
                    "warehouse": wh,
                    "quantity": qty,
                    "reorder_point": reorder,
                    "days_of_supply": round(qty / daily_sales, 1)
                    if daily_sales > 0
                    else 999,
                }
            )
    return inventory


def _generate_abandoned_carts(
    rng: random.Random,
    products: list[dict[str, Any]],
    customers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    carts: list[dict[str, Any]] = []
    for i in range(1, 16):
        customer = rng.choice(customers)
        num_items = rng.randint(1, 3)
        items_products = rng.sample(products, k=min(num_items, len(products)))
        items = []
        for p in items_products:
            qty = rng.randint(1, 2)
            items.append(
                {
                    "product_id": p["id"],
                    "product_name": p["name"],
                    "quantity": qty,
                    "unit_price": p["price"],
                }
            )
        total = round(sum(it["unit_price"] * it["quantity"] for it in items), 2)
        created = _random_date(rng, datetime(2025, 10, 1), datetime(2025, 12, 31))
        carts.append(
            {
                "id": _uid("CART", i),
                "customer_id": customer["id"],
                "customer_email": customer["email"],
                "items": items,
                "created_at": _iso_date(created),
                "total_value": total,
            }
        )
    return carts


def _generate_promotions(
    rng: random.Random, products: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    promos: list[dict[str, Any]] = []
    templates = [
        ("Summer Electronics Sale", "percentage", 15.0, ["Electronics"]),
        ("Kitchen Essentials Deal", "fixed_amount", 10.0, ["Home & Kitchen"]),
        ("New Customer Welcome", "percentage", 20.0, []),
        ("BOGO Fitness Gear", "buy_one_get_one", 0.0, ["Sports & Outdoors"]),
        ("Holiday Fashion Blitz", "percentage", 25.0, ["Clothing"]),
    ]
    for i, (name, dtype, val, cats) in enumerate(templates, 1):
        if cats:
            promo_products = [
                p["id"]
                for p in products
                if any(p["category"].startswith(c) for c in cats)
            ]
        else:
            promo_products = [
                p["id"] for p in rng.sample(products, k=min(5, len(products)))
            ]
        start = _random_date(rng, datetime(2025, 6, 1), datetime(2025, 9, 1))
        end = start + timedelta(days=rng.randint(7, 30))
        promos.append(
            {
                "id": _uid("PROMO", i),
                "name": name,
                "product_ids": promo_products[:15],
                "discount_type": dtype,
                "discount_value": val,
                "start_date": _iso_date(start),
                "end_date": _iso_date(end),
                "active": rng.choice([True, False]),
                "conditions": {
                    "customer_segment": rng.choice(["all", "vip", "new", "returning"]),
                },
            }
        )
    return promos


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------
@dataclass
class DataStore:
    """Container for all seed data."""

    products: list[dict[str, Any]] = field(default_factory=list)
    customers: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    inventory: list[dict[str, Any]] = field(default_factory=list)
    abandoned_carts: list[dict[str, Any]] = field(default_factory=list)
    promotions: list[dict[str, Any]] = field(default_factory=list)


def create_data_store(seed: int = SEED) -> DataStore:
    """Create a fully populated DataStore with deterministic data."""
    rng = random.Random(seed)
    products = _generate_products(rng)
    customers = _generate_customers(rng)
    orders = _generate_orders(rng, products, customers)
    inventory = _generate_inventory(rng, products)
    carts = _generate_abandoned_carts(rng, products, customers)
    promotions = _generate_promotions(rng, products)
    return DataStore(
        products=products,
        customers=customers,
        orders=orders,
        inventory=inventory,
        abandoned_carts=carts,
        promotions=promotions,
    )


# Quick sanity check when run directly
if __name__ == "__main__":
    store = create_data_store()
    print(f"Products:        {len(store.products)}")
    print(f"Customers:       {len(store.customers)}")
    print(f"Orders:          {len(store.orders)}")
    print(f"Inventory rows:  {len(store.inventory)}")
    print(f"Abandoned carts: {len(store.abandoned_carts)}")
    print(f"Promotions:      {len(store.promotions)}")
    print()
    print("Sample product:", store.products[0])
    print("Categories:", sorted({p["category"] for p in store.products}))
