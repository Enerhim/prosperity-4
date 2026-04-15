from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import List, Any
import json


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Trader:
    LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    EMERALD_FAIR = 10000

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.LIMITS.get(product, 80)

            # ── EMERALDS ──────────────────────────────────────────────────────
            # Market always: bid=9992, ask=10008 (spread=16, fair=10000)
            # Strategy: quote at 9993/10007 to get QUEUE PRIORITY over market makers
            # → Anyone crossing the spread fills US first (not the 9992/10008 bots)
            # → Profit = 10007 - 9993 = 14 per round trip (7 per side)
            # vs old strategy at 9999/10001 = 2 per round trip
            if product == "EMERALDS":
                fair = self.EMERALD_FAIR

                # STEP 1: Take any mispriced orders (rare but free money)
                if order_depth.sell_orders:
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if ask_price < fair:
                            available = -order_depth.sell_orders[ask_price]
                            room = limit - position
                            qty = min(available, room)
                            if qty > 0:
                                orders.append(Order(product, ask_price, qty))
                                position += qty
                        else:
                            break

                if order_depth.buy_orders:
                    for bid_price in sorted(
                        order_depth.buy_orders.keys(), reverse=True
                    ):
                        if bid_price > fair:
                            available = order_depth.buy_orders[bid_price]
                            room = limit + position
                            qty = min(available, room)
                            if qty > 0:
                                orders.append(Order(product, bid_price, -qty))
                                position -= qty
                        else:
                            break

                # STEP 2: Quote just inside best bid/ask for queue priority
                # Get actual best bid/ask from order book
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())

                    # Quote one tick inside each → we jump to front of queue
                    our_bid = best_bid + 1  # e.g. 9993 (beats 9992)
                    our_ask = best_ask - 1  # e.g. 10007 (beats 10008)

                    # Only quote if our prices don't cross (shouldn't happen with 16-tick spread)
                    if our_bid < fair and our_ask > fair:
                        buy_room = limit - position
                        sell_room = limit + position

                        # Skew size based on position to stay balanced
                        if buy_room > 0:
                            orders.append(Order(product, our_bid, buy_room))
                        if sell_room > 0:
                            orders.append(Order(product, our_ask, -sell_room))

            # ── TOMATOES ─────────────────────────────────────────────────────
            # Market always: bid~X, ask~X+13 (spread=13)
            # Strategy: quote at best_bid+1 / best_ask-1 for queue priority
            # Profit per round trip = 11 (vs spread of 13, we give away 1 each side)
            # Tomatoes price drifts → use position skewing to avoid directional exposure
            elif product == "TOMATOES":
                if not order_depth.sell_orders or not order_depth.buy_orders:
                    result[product] = orders
                    continue

                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                spread = best_ask - best_bid

                # Only quote when spread is wide enough to be worth it
                if spread >= 10:
                    our_bid = best_bid + 1
                    our_ask = best_ask - 1

                    # Ensure we have edge (our bid < our ask)
                    if our_bid < our_ask:
                        buy_room = limit - position
                        sell_room = limit + position

                        # Position-skewed sizing:
                        # If we're long → reduce buy qty, increase sell qty (offload risk)
                        # If we're short → increase buy qty, reduce sell qty
                        # This keeps us near flat and prevents inventory blowup on trends

                        # Skew factor: 1.0 = full size, 0 = nothing
                        long_skew = max(
                            0.0, 1.0 - position / limit
                        )  # reduces as we go long
                        short_skew = max(
                            0.0, 1.0 + position / limit
                        )  # reduces as we go short

                        buy_qty = int(buy_room * long_skew)
                        sell_qty = int(sell_room * short_skew)

                        if buy_qty > 0:
                            orders.append(Order(product, our_bid, buy_qty))
                        if sell_qty > 0:
                            orders.append(Order(product, our_ask, -sell_qty))

            result[product] = orders

        conversions = 0
        traderData = ""
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
