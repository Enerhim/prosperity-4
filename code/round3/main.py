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
from typing import List, Any, Dict, Optional
import json
import math
import jsonpickle
from statistics import NormalDist

def norm_cdf(x: float) -> float:
    return NormalDist().cdf(x)
 
def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-10 or S <= 0:
        return max(S - K, 0.0)
    sigma = max(sigma, 1e-6)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-10 or S <= 0:
        return 1.0 if S >= K else 0.0
    sigma = max(sigma, 1e-6)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)
 
LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":        200,
    "VELVETFRUIT_EXTRACT":  200,
    "VEV_4000": 300, "VEV_4500": 300,
    "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
    "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}
 
STRIKES: Dict[str, int] = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
    "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
 
# Round 3 live constants
VEV_SIGMA = 0.22      
VEV_TTE   = 5 / 252   

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
    @staticmethod
    def _mid(od: Optional[OrderDepth]) -> Optional[float]:
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None
 
    def _update_ewm(self, td: dict, key: str, value: float, alpha: float) -> float:
        if key not in td:
            td[key] = value
        else:
            td[key] = alpha * value + (1 - alpha) * td[key]
        return td[key]
 
    @staticmethod
    def _take_book(product: str, od: OrderDepth,
                   fair: float, edge: float,
                   buy_cap: int, sell_cap: int) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
 
        if od.sell_orders:
            for ask in sorted(od.sell_orders.keys()):
                if ask > fair - edge or buy_cap <= 0:
                    break
                vol = min(-od.sell_orders[ask], buy_cap)
                orders.append(Order(product, ask, vol))
                buy_cap -= vol
 
        if od.buy_orders:
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                if bid < fair + edge or sell_cap <= 0:
                    break
                vol = min(od.buy_orders[bid], sell_cap)
                orders.append(Order(product, bid, -vol))
                sell_cap -= vol
 
        return orders, buy_cap, sell_cap
 
    def _trade_delta1(self, product: str, od: OrderDepth, pos: int, limit: int,
                      fair: float, effective_pos: float = None) -> List[Order]:
        buy_cap  = limit - pos
        sell_cap = limit + pos
 
        skew_max = 3.0
        current_exposure = effective_pos if effective_pos is not None else pos
        skew = -skew_max * (current_exposure / limit)
        
        # Clamp skew so it doesn't push our quotes completely out of the market
        skew = max(min(skew, skew_max), -skew_max)
        adj_fair = fair + skew
 
        orders, buy_cap, sell_cap = self._take_book(
            product, od, adj_fair, edge=1.0,
            buy_cap=buy_cap, sell_cap=sell_cap
        )
 
        mm_qty   = 20
        mm_half  = 2
 
        bid_price = int(adj_fair) - mm_half
        ask_price = int(adj_fair) + mm_half
 
        if bid_price >= ask_price:
            bid_price = int(adj_fair) - 1
            ask_price = int(adj_fair) + 1
 
        if buy_cap > 0:
            orders.append(Order(product, bid_price, min(mm_qty, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_price, -min(mm_qty, sell_cap)))
 
        return orders
 
    def _trade_option(self, product: str, od: OrderDepth, pos: int, limit: int,
                      vev_fair: float, strike: int) -> List[Order]:
        fair = bs_call(vev_fair, strike, VEV_TTE, VEV_SIGMA)
 
        if fair < 0.5:
            return []
 
        if fair > 500:
            edge = max(10, fair * 0.01)
        elif fair > 50:
            edge = max(6, fair * 0.05)
        else:
            edge = max(3, fair * 0.08)
 
        buy_cap  = limit - pos
        sell_cap = limit + pos
 
        orders, buy_cap, sell_cap = self._take_book(
            product, od, fair, edge=edge,
            buy_cap=buy_cap, sell_cap=sell_cap
        )
 
        mm_qty   = 10
        bid_px   = max(1, round(fair - edge))
        ask_px   = round(fair + edge) + 1
 
        if buy_cap > 0 and bid_px > 0:
            orders.append(Order(product, bid_px, min(mm_qty, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -min(mm_qty, sell_cap)))
 
        return orders
 
    def run(self, state: TradingState):
        td: dict = {}
        if state.traderData:
            try:
                td = jsonpickle.decode(state.traderData)
            except Exception:
                td = {}
 
        result: Dict[str, List[Order]] = {}
 
        # ── Update Fair Values ─────────────────────────────────────
        # Hardcode HYDROGEL_PACK to its true mean to prevent trend-following losses
        hp_fair = 10000.0 
        
        # Slower EMA for VELVETFRUIT_EXTRACT to filter out noise 
        vev_mid = self._mid(state.order_depths.get("VELVETFRUIT_EXTRACT"))
        if vev_mid is not None:
            vev_fair = self._update_ewm(td, "fair_VELVETFRUIT_EXTRACT", vev_mid, 0.005)
        else:
            vev_fair = td.get("fair_VELVETFRUIT_EXTRACT", 5252.0)

        # ── Calculate Partial Portfolio Delta ───────────────────────
        total_velvet_delta = float(state.position.get("VELVETFRUIT_EXTRACT", 0))
        for product, strike in STRIKES.items():
            pos = state.position.get(product, 0)
            if pos != 0:
                opt_delta = bs_call_delta(vev_fair, strike, VEV_TTE, VEV_SIGMA)
                # Scale down options impact by 0.3 so hedging doesn't eat position limits
                total_velvet_delta += (pos * opt_delta) * 0.3 
 
        # ── Route Orders ──────────────────────────────────────────
        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = LIMITS.get(product, 100)
 
            if product == "HYDROGEL_PACK":
                orders = self._trade_delta1(product, od, pos, limit, hp_fair)
 
            elif product == "VELVETFRUIT_EXTRACT":
                orders = self._trade_delta1(product, od, pos, limit, vev_fair, effective_pos=total_velvet_delta)
 
            elif product in STRIKES:
                orders = self._trade_option(
                    product, od, pos, limit,
                    vev_fair=vev_fair,
                    strike=STRIKES[product]
                )
            else:
                orders = []
 
            result[product] = orders
 
        logger.flush(state, result, 0, jsonpickle.encode(td))
        return result, 0, jsonpickle.encode(td)
