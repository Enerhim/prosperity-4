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
import math
import jsonpickle


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
    IPR = "INTARIAN_PEPPER_ROOT"
    ACO = "ASH_COATED_OSMIUM"
    LIMIT = 80
    ACO_ALPHA = 0.5

    ACO_BB_PERIOD = 20  # rolling window for mean & std
    ACO_BB_WIDTH = 2.5  # band width in std-devs (classic setting)
    ACO_BB_TIGHT = 1.0  # inner band: start leaning at ±0.5σ

    def run(self, state: TradingState):
        td = {}
        if state.traderData:
            try:
                td = jsonpickle.decode(state.traderData)
            except:
                td = {}

        result = {}
        ipr = self._trade_pepper(state)
        if ipr:
            result[self.IPR] = ipr
        aco = self._trade_osmium(state, td)
        if aco:
            result[self.ACO] = aco

        logger.flush(state, result, 0, jsonpickle.encode(td))
        return result, 0, jsonpickle.encode(td)

    def _trade_pepper(self, state):
        """Buy max position ASAP. IPR price rises +1/tick = +1000/day."""
        od = state.order_depths.get(self.IPR)
        if not od:
            return []
        pos = state.position.get(self.IPR, 0)
        room = self.LIMIT - pos
        if room <= 0:
            return []

        orders = []
        remaining = room
        for price in sorted(od.sell_orders):
            qty = min(abs(od.sell_orders[price]), remaining)
            orders.append(Order(self.IPR, price, qty))
            remaining -= qty
            if remaining <= 0:
                break

        # Post aggressive bid (best_bid+1) to fill remaining fast
        if remaining > 0 and od.buy_orders:
            orders.append(Order(self.IPR, max(od.buy_orders) + 1, remaining))

        return orders

    def _trade_osmium(self, state, td):
        """
        Bollinger-Band fair value for ACO.
        - Rolling mean = fair value anchor.
        - Price near lower band → lean long  (bid more aggressively).
        - Price near upper band → lean short (ask more aggressively).
        - Price outside band   → aggressive take (mean-reversion bet).
        Welford online algorithm keeps the rolling mean/variance in O(1).
        """
        od = state.order_depths.get(self.ACO)
        if not od:
            return []
        pos = state.position.get(self.ACO, 0)
        mid = self._mid(od)
        if mid is None:
            return []

        # ── Welford online mean/variance over last BB_PERIOD ticks ────────────────
        history = td.get("bb_history", [])
        history.append(mid)
        if len(history) > self.ACO_BB_PERIOD:
            history.pop(0)
        td["bb_history"] = history

        n = len(history)
        mean = sum(history) / n
        variance = sum((x - mean) ** 2 for x in history) / n if n > 1 else 0.0
        std = variance**0.5

        # ── Fair = rolling mean; bands as reference ────────────────────────────────
        upper = mean + self.ACO_BB_WIDTH * std
        lower = mean - self.ACO_BB_WIDTH * std
        inner_upper = mean + self.ACO_BB_TIGHT * std
        inner_lower = mean - self.ACO_BB_TIGHT * std

        fair = mean
        fair_r = round(fair)

        buy_cap = self.LIMIT - pos
        sell_cap = self.LIMIT + pos
        orders = []

        # ── Aggressive take: price outside outer bands (mean-reversion) ───────────
        if std > 0:
            if od.sell_orders and buy_cap > 0 and mid < lower:
                ba = min(od.sell_orders)
                qty = min(abs(od.sell_orders[ba]), buy_cap)
                orders.append(Order(self.ACO, ba, qty))
                buy_cap -= qty

            if od.buy_orders and sell_cap > 0 and mid > upper:
                bb = max(od.buy_orders)
                qty = min(abs(od.buy_orders[bb]), sell_cap)
                orders.append(Order(self.ACO, bb, -qty))
                sell_cap -= qty

        # ── Passive MM: skew quotes toward mean based on band position ─────────────
        inv_skew = pos // 8

        # +1 → lean to sell (price high in band), -1 → lean to buy (price low)
        if std > 0 and mid > inner_upper:
            bb_skew = 1
        elif std > 0 and mid < inner_lower:
            bb_skew = -1
        else:
            bb_skew = 0

        skew = inv_skew + bb_skew

        if buy_cap > 0 and od.buy_orders:
            bb = max(od.buy_orders)
            bid = (bb + 1) - max(0, skew)
            bid = min(bid, fair_r - 1)
            if bid >= fair_r - 10:
                orders.append(Order(self.ACO, bid, buy_cap))

        if sell_cap > 0 and od.sell_orders:
            ba = min(od.sell_orders)
            ask = (ba - 1) - min(0, skew)
            ask = max(ask, fair_r + 1)
            if ask <= fair_r + 10:
                orders.append(Order(self.ACO, ask, -sell_cap))

        return orders

    @staticmethod
    def _mid(od):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2
        if od.buy_orders:
            return float(max(od.buy_orders))
        if od.sell_orders:
            return float(min(od.sell_orders))
        return None
