"""
documents.py — Daily Essentials' company knowledge base (mock).

Stand-in for the company_documents/*.pdf folder from Lecture 4. Each
entry is one "document"; rag.py splits these into paragraph-level
chunks and indexes them. Swap this for a real PDF loader later
without touching rag.py's chunking/embedding/search functions.

Note the return_policy.pdf / warranty_policy.pdf split below is
deliberate: it reproduces the exact Lecture 4 example ("mixer bought
in the Diwali sale, stopped working after 20 days") — 20 days is past
the 15-day electronics return window, but well within the one-year
warranty, so the right answer requires reading *both* documents.
"""

POLICY_DOCUMENTS = [
    {
        "source": "return_policy.pdf",
        "text": """
General return policy: Most items purchased from Daily Essentials may be returned within 30 days of delivery for a full refund, provided the item is unused and in its original packaging.

Electronics: Electronics (including kitchen appliances such as mixers, blenders, and coffee makers) may be returned within 15 days of delivery, not 30. This shorter window applies whether or not the item is defective.

Clothing and textiles: Clothing, linens, and textile items may be returned within 30 days, and do not need to be in original packaging as long as tags are still attached.

Damaged or defective products: Items that arrive damaged, or that stop working after the standard return window has passed, are not eligible for a standard return — see warranty_policy.pdf for repair and replacement options instead.
""",
    },
    {
        "source": "refund_policy.pdf",
        "text": """
Refund method: Approved refunds are issued to the original payment method within 5-7 business days of Daily Essentials receiving the returned item.

Store credit option: Customers may choose store credit instead of a refund to the original payment method; store credit is issued immediately and never expires.

Shipping costs: Original shipping charges are non-refundable unless the return is due to a Daily Essentials error (wrong item shipped, item arrived damaged).
""",
    },
    {
        "source": "shipping_policy.pdf",
        "text": """
Standard shipping: Orders are normally delivered within 3-5 business days of being placed.

Festival sale shipping: During major sale events (Diwali Sale, End of Season Sale), delivery times may extend to 7-10 business days due to higher order volumes.

Tracking: A tracking number is emailed once an order ships, and can also be looked up in the Daily Essentials app under 'My Orders'.
""",
    },
    {
        "source": "warranty_policy.pdf",
        "text": """
Standard warranty: Electrical appliances, including mixers, blenders, and coffee makers, carry a one-year manufacturer's warranty against defects in materials and workmanship.

Warranty claims after delivery: If an appliance stops working within the one-year warranty period, the customer is entitled to a free repair or replacement, regardless of whether the standard return window (see return_policy.pdf) has already passed.

What voids the warranty: Warranty coverage does not apply to damage caused by misuse, unauthorized repairs, or normal wear on consumable parts (such as blender blades).

Festival sale items: Items purchased during a festival sale carry the same one-year warranty as items purchased at full price — the sale price does not shorten the warranty period.
""",
    },
    {
        "source": "exchange_policy.pdf",
        "text": """
Size and color exchanges: Clothing items can be exchanged for a different size or color within 30 days of delivery, subject to availability.

Exchanging electronics: Electronics are not eligible for a direct exchange; customers should follow the return process instead and place a new order.
""",
    },
]