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
    # Bollinger Bands
    ACO_BB_PERIOD = 20
    ACO_BB_WIDTH = 2.5
    ACO_BB_TIGHT = 1.0

    # RSI
    ACO_RSI_PERIOD = 20  # lookback for avg gain/loss
    ACO_RSI_OB = 62  # overbought threshold → lean short
    ACO_RSI_OS = 38  # oversold threshold  → lean long
    ACO_RSI_EXTREME_OB = 78  # aggressive sell signal
    ACO_RSI_EXTREME_OS = 22  # aggressive buy  signal

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
        Hybrid BB + RSI for ACO fair value.

        Signal logic:
        STRONG BUY  → mid below BB lower band  AND RSI oversold  (both agree)
        STRONG SELL → mid above BB upper band  AND RSI overbought (both agree)
        SOFT   BUY  → only one of the two signals firing → lean, don't take
        NEUTRAL     → neither signal → pure inventory-skewed MM

        Fair value = BB mean, biased by RSI toward the mean-reversion target.
        """
        od = state.order_depths.get(self.ACO)
        if not od:
            return []
        pos = state.position.get(self.ACO, 0)
        mid = self._mid(od)
        if mid is None:
            return []

        # ── Bollinger Bands (rolling window) ──────────────────────────────────────
        history = td.get("bb_history", [])
        history.append(mid)
        if len(history) > self.ACO_BB_PERIOD:
            history.pop(0)
        td["bb_history"] = history

        n = len(history)
        bb_mean = sum(history) / n
        variance = sum((x - bb_mean) ** 2 for x in history) / n if n > 1 else 0.0
        std = variance**0.5

        upper = bb_mean + self.ACO_BB_WIDTH * std
        lower = bb_mean - self.ACO_BB_WIDTH * std
        inner_upper = bb_mean + self.ACO_BB_TIGHT * std
        inner_lower = bb_mean - self.ACO_BB_TIGHT * std

        # ── RSI (Wilder smoothing) ─────────────────────────────────────────────────
        prev_mid = td.get("rsi_prev_mid", mid)
        avg_gain = td.get("rsi_avg_gain", 0.0)
        avg_loss = td.get("rsi_avg_loss", 0.0)
        tick_count = td.get("rsi_ticks", 0) + 1

        delta = mid - prev_mid
        alpha = 1.0 / self.ACO_RSI_PERIOD
        avg_gain = (1 - alpha) * avg_gain + alpha * max(delta, 0.0)
        avg_loss = (1 - alpha) * avg_loss + alpha * max(-delta, 0.0)

        if avg_loss > 0:
            rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        elif avg_gain > 0:
            rsi = 100.0
        else:
            rsi = 50.0

        td["rsi_prev_mid"] = mid
        td["rsi_avg_gain"] = avg_gain
        td["rsi_avg_loss"] = avg_loss
        td["rsi_ticks"] = tick_count

        # ── Combined signal ────────────────────────────────────────────────────────
        bb_buy = std > 0 and mid < lower
        bb_sell = std > 0 and mid > upper
        bb_soft_buy = std > 0 and mid < inner_lower
        bb_soft_sell = std > 0 and mid > inner_upper

        rsi_buy = rsi <= self.ACO_RSI_EXTREME_OS
        rsi_sell = rsi >= self.ACO_RSI_EXTREME_OB
        rsi_soft_buy = rsi < self.ACO_RSI_OS
        rsi_soft_sell = rsi > self.ACO_RSI_OB

        strong_buy = bb_buy and rsi_buy  # both scream oversold  → aggressive
        strong_sell = bb_sell and rsi_sell  # both scream overbought → aggressive
        soft_buy = (bb_soft_buy or rsi_soft_buy) and not strong_sell
        soft_sell = (bb_soft_sell or rsi_soft_sell) and not strong_buy

        # ── Fair value: BB mean + small RSI pull ──────────────────────────────────
        # RSI bias nudges fair toward where mean-reversion would take us
        rsi_bias = (50.0 - rsi) / 25.0  # ±2.5 ticks max
        fair = bb_mean + rsi_bias
        fair_r = round(fair)

        buy_cap = self.LIMIT - pos
        sell_cap = self.LIMIT + pos
        orders = []

        # ── Aggressive take: STRONG signal only ───────────────────────────────────
        if strong_buy and od.sell_orders and buy_cap > 0:
            ba = min(od.sell_orders)
            qty = min(abs(od.sell_orders[ba]), buy_cap)
            orders.append(Order(self.ACO, ba, qty))
            buy_cap -= qty

        if strong_sell and od.buy_orders and sell_cap > 0:
            bb = max(od.buy_orders)
            qty = min(abs(od.buy_orders[bb]), sell_cap)
            orders.append(Order(self.ACO, bb, -qty))
            sell_cap -= qty

        # ── Passive MM: skew from inventory + combined signal ─────────────────────
        inv_skew = pos // 12

        if strong_sell or soft_sell:
            sig_skew = 2 if strong_sell else 1  # stronger signal = harder skew
        elif strong_buy or soft_buy:
            sig_skew = -2 if strong_buy else -1
        else:
            sig_skew = 0

        skew = inv_skew + sig_skew

        if buy_cap > 0 and od.buy_orders:
            bb_price = max(od.buy_orders)
            bid = (bb_price + 1) - max(0, skew)
            bid = min(bid, fair_r - 1)
            if bid >= fair_r - 10:
                orders.append(Order(self.ACO, bid, buy_cap))

        if sell_cap > 0 and od.sell_orders:
            ba_price = min(od.sell_orders)
            ask = (ba_price - 1) - min(0, skew)
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
