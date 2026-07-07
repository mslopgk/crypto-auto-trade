"""Order execution brokers.

Two implementations behind one ABC:

- :class:`PaperBroker` — in-memory fill engine (fee + slippage cost model)
  fed by an injectable ``price_source``; used for paper trading and tests.
- :class:`CcxtBroker` — thin, defensive wrapper around a synchronous ccxt
  client implementing the order policy from docs/research-brief.md §4.2:
  precision/min-notional pre-checks, uuid clientOrderId, reconcile-by-client-id
  on timeouts (never blind re-place), tiered retry/backoff per error class.

All quantities follow ccxt semantics: market **buys** are specified by quote
``cost``, market **sells** by base ``amount``.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable

import ccxt

log = logging.getLogger(__name__)

# order-placement retry policy (research brief §4.2)
_MAX_ORDER_TRIES = 8
_MAX_BACKOFF_S = 60.0
_RATE_LIMIT_PAUSE_S = 60.0

_CLIENT_ID_PARAM = {
    "binance": "newClientOrderId",
    "binanceusdm": "newClientOrderId",
    "binancecoinm": "newClientOrderId",
    "upbit": "identifier",
}
_CLIENT_ID_QUERY_PARAM = {
    "binance": "origClientOrderId",
    "binanceusdm": "origClientOrderId",
    "binancecoinm": "origClientOrderId",
    "upbit": "identifier",
}


class BrokerError(Exception):
    """Order could not be placed / reconciled."""


class BrokerAuthError(BrokerError):
    """Authentication failed — never retried; engine must halt and alert."""


@dataclass
class Fill:
    """Normalized result of an executed market order."""

    symbol: str
    side: str          # 'buy' | 'sell'
    amount: float      # base units filled
    price: float       # average fill price
    cost: float        # quote notional (amount * price)
    fee: float         # fee in quote currency (estimated if not reported)
    timestamp: float   # unix seconds, UTC
    order_id: str

    def to_dict(self) -> dict:
        return asdict(self)


class Broker(ABC):
    """Execution venue abstraction used by the live engine."""

    @abstractmethod
    def market_order(self, symbol: str, side: str, amount: float | None = None,
                     cost: float | None = None) -> Fill:
        """Place a market order. Buys take ``cost`` (quote), sells ``amount`` (base)."""

    @abstractmethod
    def get_balances(self) -> dict[str, float]:
        """Currency code -> total balance."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        """Most recent trade/mark price for ``symbol``."""

    def close(self) -> None:
        """Release any underlying resources. Idempotent."""


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------

class PaperBroker(Broker):
    """In-memory paper fill engine.

    Fills at ``price_source(symbol)`` adjusted adversely by ``slippage``
    (buys ``*(1+slip)``, sells ``*(1-slip)``); ``fee`` charged on notional.
    Buy semantics mirror the backtest engine: ``cost`` is the TOTAL quote
    spent including fee, so ``amount = cost / (px * (1 + fee))``.
    Thread-safe.
    """

    def __init__(self, quote_balance: float, fee: float = 0.0005,
                 slippage: float = 0.0005,
                 price_source: Callable[[str], float] | None = None,
                 quote_currency: str = "USDT"):
        self._fee = float(fee)
        self._slip = float(slippage)
        self._price_source = price_source
        self._quote_ccy = quote_currency
        self._quote = float(quote_balance)
        self._bases: dict[str, float] = {}
        self._lock = threading.Lock()

    # -- Broker API ----------------------------------------------------------
    def market_order(self, symbol: str, side: str, amount: float | None = None,
                     cost: float | None = None) -> Fill:
        if side not in ("buy", "sell"):
            raise ValueError(f"invalid side {side!r}")
        base = symbol.split("/")[0]
        px = self.get_last_price(symbol)
        with self._lock:
            if side == "buy":
                fill_px = px * (1.0 + self._slip)
                if cost is None:
                    if amount is None:
                        raise ValueError("market buy requires cost (quote) or amount (base)")
                    cost = amount * fill_px * (1.0 + self._fee)
                cost = float(cost)
                if cost <= 0.0:
                    raise ValueError("market buy cost must be positive")
                if cost > self._quote * (1.0 + 1e-9):
                    raise ValueError(
                        f"insufficient funds: buy cost {cost:.8g} > "
                        f"{self._quote_ccy} balance {self._quote:.8g}")
                units = cost / (fill_px * (1.0 + self._fee))
                fee_paid = units * fill_px * self._fee
                self._quote -= units * fill_px + fee_paid
                self._bases[base] = self._bases.get(base, 0.0) + units
                fill = Fill(symbol, "buy", units, fill_px, units * fill_px,
                            fee_paid, time.time(), f"paper-{uuid.uuid4().hex[:12]}")
            else:
                if amount is None:
                    raise ValueError("market sell requires amount (base)")
                amount = float(amount)
                held = self._bases.get(base, 0.0)
                if amount <= 0.0:
                    raise ValueError("market sell amount must be positive")
                if amount > held * (1.0 + 1e-9):
                    raise ValueError(
                        f"insufficient funds: sell {amount:.8g} {base} > held {held:.8g}")
                amount = min(amount, held)
                fill_px = px * (1.0 - self._slip)
                proceeds = amount * fill_px
                fee_paid = proceeds * self._fee
                self._quote += proceeds - fee_paid
                self._bases[base] = held - amount
                fill = Fill(symbol, "sell", amount, fill_px, proceeds,
                            fee_paid, time.time(), f"paper-{uuid.uuid4().hex[:12]}")
        log.debug("paper fill: %s", fill)
        return fill

    def get_balances(self) -> dict[str, float]:
        with self._lock:
            out = {self._quote_ccy: self._quote}
            out.update(self._bases)
            return out

    def get_last_price(self, symbol: str) -> float:
        if self._price_source is None:
            raise BrokerError("PaperBroker has no price_source configured")
        px = self._price_source(symbol)
        if px is None or not math.isfinite(px) or px <= 0:
            raise BrokerError(f"PaperBroker has no price yet for {symbol}")
        return float(px)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ccxt broker
# ---------------------------------------------------------------------------

class CcxtBroker(Broker):
    """Live broker over a synchronous ccxt client.

    Markets are loaded lazily once. Order placement runs through a single
    code path with precision + min-notional pre-checks and the error policy
    from the research brief (§4.2). ``RequestTimeout`` triggers reconciliation
    by clientOrderId before any re-place.
    """

    def __init__(self, exchange_id: str, api_key: str, secret: str,
                 sandbox: bool = False):
        self._exchange_id = exchange_id
        klass = getattr(ccxt, exchange_id)
        self._ex: ccxt.Exchange = klass({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if sandbox:
            self._ex.set_sandbox_mode(True)
        self._markets_loaded = False
        self._lock = threading.Lock()
        self._last_prices: dict[str, float] = {}

    # -- internals -----------------------------------------------------------
    def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            self._retry(self._ex.load_markets)
            self._markets_loaded = True

    def _client_id_param(self) -> str:
        return _CLIENT_ID_PARAM.get(self._exchange_id, "clientOrderId")

    def _client_id_query_param(self) -> str:
        return _CLIENT_ID_QUERY_PARAM.get(self._exchange_id, "clientOrderId")

    def _min_notional(self, symbol: str) -> float:
        """Minimum order notional in quote currency (market limits or fallback)."""
        market = self._ex.market(symbol)
        mn = ((market.get("limits") or {}).get("cost") or {}).get("min")
        if mn:
            return float(mn)
        quote = market.get("quote") or symbol.split("/")[-1]
        return 5000.0 if quote == "KRW" else 5.0

    def _check_min_notional(self, symbol: str, notional: float) -> None:
        mn = self._min_notional(symbol)
        if notional < mn:
            raise ValueError(
                f"order notional {notional:.8g} below min {mn:.8g} for {symbol}")

    def _retry(self, fn: Callable, tries: int = 5):
        """Retry wrapper for idempotent READ calls (never used for placement)."""
        backoff = 1.0
        for attempt in range(tries):
            try:
                return fn()
            except ccxt.AuthenticationError as e:
                raise BrokerAuthError(str(e)) from e
            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
                if attempt == tries - 1:
                    raise
                log.warning("rate limited (%s); pausing %.0fs", e, _RATE_LIMIT_PAUSE_S)
                time.sleep(_RATE_LIMIT_PAUSE_S)
            except ccxt.NetworkError as e:
                if attempt == tries - 1:
                    raise
                wait = min(backoff, _MAX_BACKOFF_S) + random.uniform(0, backoff * 0.25)
                log.warning("network error (%s); retry in %.1fs", e, wait)
                time.sleep(wait)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)

    def _upbit_market_buy(self, symbol: str, cost: float, params: dict) -> dict:
        """Upbit market buy: cost passed as the amount argument.

        Per ccxt upbit convention, with createMarketBuyOrderRequiresPrice=False
        createOrder(symbol, 'market', 'buy', amount) treats ``amount`` as the
        KRW cost (Upbit native 'price' order type).
        """
        self._ex.options["createMarketBuyOrderRequiresPrice"] = False
        amount_arg = self._ex.cost_to_precision(symbol, cost)
        return self._ex.create_order(symbol, "market", "buy", amount_arg, None, params)

    def _place(self, symbol: str, side: str, amount: float | None,
               cost: float | None, params: dict) -> dict:
        if side == "buy":
            if cost is None:
                if amount is None:
                    raise ValueError("market buy requires cost (quote) or amount (base)")
                cost = float(amount) * self.get_last_price(symbol)
            cost = float(cost)
            self._check_min_notional(symbol, cost)
            if self._exchange_id == "upbit":
                return self._upbit_market_buy(symbol, cost, params)
            if self._ex.has.get("createMarketBuyOrderWithCost"):
                return self._ex.create_market_buy_order_with_cost(symbol, cost, params)
            # generic fallback: convert cost to base amount at last price
            px = self.get_last_price(symbol)
            amt = float(self._ex.amount_to_precision(symbol, cost / px))
            return self._ex.create_order(symbol, "market", "buy", amt, None, params)
        # sell
        if amount is None:
            raise ValueError("market sell requires amount (base)")
        amt = float(self._ex.amount_to_precision(symbol, float(amount)))
        if amt <= 0.0:
            raise ValueError(f"sell amount rounds to zero for {symbol}")
        last = self._last_prices.get(symbol)
        if last:  # pre-check only when a recent price is already cached
            self._check_min_notional(symbol, amt * last)
        return self._ex.create_order(symbol, "market", "sell", amt, None, params)

    def _find_order_by_client_id(self, symbol: str, client_id: str) -> dict | None:
        """Reconcile after a timeout: did the order reach the exchange?"""
        q = self._client_id_query_param()
        try:
            order = self._ex.fetch_order(None, symbol, params={q: client_id})
            if order:
                return order
        except ccxt.OrderNotFound:
            return None
        except Exception as e:
            log.warning("fetch_order by client id failed (%s); scanning recent orders", e)
        try:
            orders = self._ex.fetch_orders(symbol, limit=20)
        except Exception as e:
            raise BrokerError(
                f"cannot reconcile order {client_id} after timeout: {e}") from e
        for o in orders or []:
            info = o.get("info") or {}
            if o.get("clientOrderId") == client_id or client_id in (
                    info.get("clientOrderId"), info.get("origClientOrderId"),
                    info.get("identifier")):
                return o
        return None

    def _await_settlement(self, order: dict, symbol: str, polls: int = 5) -> dict:
        """Poll a just-placed market order until it reports a fill."""
        for _ in range(polls):
            filled = order.get("filled")
            status = order.get("status")
            if (filled and filled > 0) or status == "closed":
                return order
            time.sleep(1.0)
            try:
                order = self._ex.fetch_order(order.get("id"), symbol) or order
            except Exception as e:  # keep last known snapshot
                log.warning("settlement poll failed: %s", e)
                break
        return order

    def _order_to_fill(self, order: dict, symbol: str, side: str, client_id: str) -> Fill:
        market = self._ex.market(symbol)
        amount = float(order.get("filled") or order.get("amount") or 0.0)
        price = order.get("average") or order.get("price")
        cost = order.get("cost")
        if not price and cost and amount:
            price = float(cost) / amount
        price = float(price or self._last_prices.get(symbol) or 0.0)
        cost = float(cost) if cost else amount * price
        fee = self._extract_fee(order, market, price, cost)
        ts_ms = order.get("timestamp")
        ts = float(ts_ms) / 1000.0 if ts_ms else time.time()
        return Fill(symbol=symbol, side=side, amount=amount, price=price,
                    cost=cost, fee=fee, timestamp=ts,
                    order_id=str(order.get("id") or client_id))

    def _extract_fee(self, order: dict, market: dict, price: float, cost: float) -> float:
        """Fee in quote currency from the order response, else cost-model estimate."""
        entries = []
        if order.get("fee"):
            entries = [order["fee"]]
        elif order.get("fees"):
            entries = order["fees"]
        total = 0.0
        seen = False
        for f in entries:
            fc = f.get("cost")
            if fc is None:
                continue
            seen = True
            # fees charged in base currency (e.g. Binance buys) -> convert to quote
            if f.get("currency") == market.get("base") and price > 0:
                total += float(fc) * price
            else:
                total += float(fc)
        if seen:
            return total
        taker = market.get("taker")
        return cost * float(taker if taker is not None else 0.001)

    # -- Broker API ----------------------------------------------------------
    def market_order(self, symbol: str, side: str, amount: float | None = None,
                     cost: float | None = None) -> Fill:
        if side not in ("buy", "sell"):
            raise ValueError(f"invalid side {side!r}")
        with self._lock:
            self._ensure_markets()
            client_id = uuid.uuid4().hex
            params = {self._client_id_param(): client_id}
            backoff = 1.0
            order: dict | None = None
            for attempt in range(1, _MAX_ORDER_TRIES + 1):
                try:
                    order = self._place(symbol, side, amount, cost, params)
                    break
                except ccxt.AuthenticationError as e:
                    raise BrokerAuthError(str(e)) from e
                except (ccxt.InvalidOrder, ccxt.InsufficientFunds):
                    raise  # never retry unchanged
                except ccxt.RequestTimeout as e:
                    # order may have reached the exchange: reconcile, NEVER blind re-place
                    log.warning("order timeout (%s); reconciling by clientOrderId", e)
                    found = self._find_order_by_client_id(symbol, client_id)
                    if found is not None:
                        order = found
                        break
                    if attempt == _MAX_ORDER_TRIES:
                        raise BrokerError(
                            f"order timed out and not found after reconcile: {e}") from e
                except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
                    if attempt == _MAX_ORDER_TRIES:
                        raise BrokerError(f"rate limited placing order: {e}") from e
                    log.warning("rate limited (%s); pausing %.0fs", e, _RATE_LIMIT_PAUSE_S)
                    time.sleep(max(_RATE_LIMIT_PAUSE_S, backoff))
                except ccxt.NetworkError as e:
                    if attempt == _MAX_ORDER_TRIES:
                        raise BrokerError(f"network failure placing order: {e}") from e
                    wait = min(backoff, _MAX_BACKOFF_S) + random.uniform(0, backoff * 0.25)
                    log.warning("network error (%s); retry %d in %.1fs", e, attempt, wait)
                    time.sleep(wait)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
            if order is None:
                raise BrokerError("order placement exhausted retries")
            order = self._await_settlement(order, symbol)
            fill = self._order_to_fill(order, symbol, side, client_id)
        log.info("live fill: %s", fill)
        return fill

    def get_balances(self) -> dict[str, float]:
        self._ensure_markets()
        bal = self._retry(self._ex.fetch_balance)
        total = bal.get("total") or {}
        return {k: float(v) for k, v in total.items() if v is not None}

    def get_last_price(self, symbol: str) -> float:
        self._ensure_markets()
        ticker = self._retry(lambda: self._ex.fetch_ticker(symbol))
        px = ticker.get("last") or ticker.get("close")
        if px is None:
            raise BrokerError(f"no last price in ticker for {symbol}")
        px = float(px)
        self._last_prices[symbol] = px
        return px

    def close(self) -> None:
        session = getattr(self._ex, "session", None)
        if session is not None:
            try:
                session.close()
            except Exception:  # best-effort resource release
                pass
