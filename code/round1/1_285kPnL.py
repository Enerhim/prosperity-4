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
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    OSMIUM_FAIR = 10000
    PEPPER_SLOPE = 0.001

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except:
            data = {}

        pepper_anchor_t = data.get("pepper_anchor_t", None)
        pepper_anchor_mid = data.get("pepper_anchor_mid", None)

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            limit = self.LIMITS[product]
            orders: List[Order] = []

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None

            # ================= OSMIUM =================
            if product == "ASH_COATED_OSMIUM":
                fair = self.OSMIUM_FAIR

                # aggressive take
                if od.sell_orders:
                    for ask in sorted(od.sell_orders):
                        if ask >= fair - 8:
                            break
                        vol = min(-od.sell_orders[ask], limit - pos)
                        if vol > 0:
                            orders.append(Order(product, ask, vol))
                            pos += vol

                if od.buy_orders:
                    for bid in sorted(od.buy_orders, reverse=True):
                        if bid <= fair + 8:
                            break
                        vol = min(od.buy_orders[bid], limit + pos)
                        if vol > 0:
                            orders.append(Order(product, bid, -vol))
                            pos -= vol

                # passive MM
                if best_bid and best_ask:
                    skew = pos / limit
                    bid = best_bid
                    ask = best_ask

                    size = 20
                    buy_qty = min(size, limit - pos)
                    sell_qty = min(size, limit + pos)

                    if buy_qty > 0:
                        orders.append(Order(product, bid, buy_qty))

                    if sell_qty > 0:
                        orders.append(Order(product, ask, -sell_qty))

            # ================= PEPPER =================
            if product == "INTARIAN_PEPPER_ROOT":
                mid = None
                if best_bid and best_ask:
                    mid = (best_bid + best_ask) / 2
                elif best_bid:
                    mid = best_bid
                elif best_ask:
                    mid = best_ask

                if mid is not None:
                    if pepper_anchor_t is None:
                        pepper_anchor_t = state.timestamp
                        pepper_anchor_mid = mid

                    fair = (
                        pepper_anchor_mid
                        + (state.timestamp - pepper_anchor_t) * self.PEPPER_SLOPE
                    )

                    # aggressive buys
                    if od.sell_orders:
                        for ask in sorted(od.sell_orders):
                            if ask > fair + 2:
                                break
                            vol = min(-od.sell_orders[ask], limit - pos)
                            if vol > 0:
                                orders.append(Order(product, ask, vol))
                                pos += vol

                    # profit taking sells
                    if od.buy_orders:
                        for bid in sorted(od.buy_orders, reverse=True):
                            if bid < fair + 4:
                                break
                            vol = min(od.buy_orders[bid], pos)
                            if vol > 0:
                                orders.append(Order(product, bid, -vol))
                                pos -= vol

                    # directional MM (long bias)
                    if best_bid:
                        size = 15
                        buy_qty = min(size, limit - pos)
                        if buy_qty > 0 and best_bid <= fair:
                            orders.append(Order(product, best_bid, buy_qty))

            result[product] = orders

        traderData = json.dumps(
            {"pepper_anchor_t": pepper_anchor_t, "pepper_anchor_mid": pepper_anchor_mid}
        )
        logger.flush(state, result, 0, traderData)
        return result, 0, traderData
