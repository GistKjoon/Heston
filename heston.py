

from __future__ import annotations
import cmath
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Callable

# Optional SciPy (for quad integration & Brent root-finding). We provide clean fallbacks.
try:
    from scipy.integrate import quad as _quad
    from scipy.optimize import brentq as _brentq
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# --------------------------
# Utilities & Numerics
# --------------------------

def _safe_quad(func: Callable[[float], float], a: float, b: float, *, limit: int = 200) -> float:
    """Numerical integration. Prefer SciPy quad, otherwise composite Simpson's rule."""
    if _HAVE_SCIPY:
        val, _ = _quad(func, a, b, limit=limit)
        return val
    # Fallback: adaptive composite Simpson
    n = 4096  # large even number for accuracy
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([func(float(xi)) for xi in x])
    S = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return float(S * h / 3.0)

def _safe_brentq(func: Callable[[float], float], a: float, b: float, *, tol: float = 1e-8, maxiter: int = 200) -> float:
    """Root-finding. Prefer SciPy Brent, otherwise bisection."""
    if _HAVE_SCIPY:
        return float(_brentq(func, a, b, xtol=tol, maxiter=maxiter))
    # Fallback: bisection
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        raise ValueError("Root not bracketed for bisection fallback.")
    lo, hi = a, b
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if abs(fm) < tol or (hi - lo) * 0.5 < tol:
            return float(mid)
        if fa * fm <= 0:
            hi = mid
            fb = fm
        else:
            lo = mid
            fa = fm
    return float(0.5 * (lo + hi))

# --------------------------
# Black–Scholes helpers
# --------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price_call_put(S0: float, K: float, T: float, r: float, q: float, sigma: float, flag: Literal["call", "put"]="call") -> float:
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        # Degenerate cases
        df_r = math.exp(-r * max(T,0.0))
        df_q = math.exp(-q * max(T,0.0))
        forward = S0 * df_q / df_r
        if flag == "call":
            return max(S0*df_q - K*df_r, 0.0)
        else:
            return max(K*df_r - S0*df_q, 0.0)

    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    F = S0 * df_q / df_r
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    if flag == "call":
        return df_r * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        return df_r * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))

def bs_implied_vol(S0: float, K: float, T: float, r: float, q: float, price: float, flag: Literal["call","put"]="call") -> float:
    # Robust IV via Brent/bisection on [1e-6, 5.0]
    def f(sig: float) -> float:
        return bs_price_call_put(S0, K, T, r, q, sig, flag) - price
    # Ensure bracket
    lo, hi = 1e-6, 5.0
    plo, phi = f(lo), f(hi)
    # If monotonicity fails (rare due to price arb), clamp
    if plo * phi > 0:
        # Try to widen hi
        hi2 = 10.0
        phi2 = f(hi2)
        if plo * phi2 > 0:
            # Fallback: return NaN
            return float("nan")
        return _safe_brentq(f, lo, hi2)
    return _safe_brentq(f, lo, hi)

# --------------------------
# Heston Model
# --------------------------

@dataclass
class HestonParams:
    S0: float = 100.0     # spot
    v0: float = 0.04      # initial variance (sigma0^2)
    kappa: float = 1.5    # mean-reversion speed
    theta: float = 0.04   # long-run variance
    xi: float = 0.5       # vol of vol
    rho: float = -0.7     # correlation between asset and variance
    r: float = 0.01       # risk-free rate
    q: float = 0.0        # dividend yield

class HestonModel:
    def __init__(self, params: HestonParams):
        self.p = params

    # Characteristic function under risk-neutral measure
    # Little Heston Trap implementation
    def charfunc(self, u: complex, T: float) -> complex:
        p = self.p
        i = 1j
        x0 = math.log(max(p.S0, 1e-15))
        a = p.kappa * p.theta
        b = p.kappa
        # Drift uses r-q
        drift = (p.r - p.q)
        d = cmath.sqrt((p.rho * p.xi * u * i - b)**2 + (p.xi**2) * (u*i + u*u))
        # "Little trap" g to avoid explosion
        g = (b - p.rho * p.xi * u * i - d) / (b - p.rho * p.xi * u * i + d)

        # Ensure numerical stability (clip |g| < 1)
        # In rare cases |g| can be exactly 1 due to rounding; nudge slightly
        if abs(g) > 0.999999:
            g = g.real * 0.999999 + 1j * g.imag * 0.0

        exp_dT = cmath.exp(-d * T)
        C = (i*u*(x0 + drift*T) +
             (a/(p.xi**2)) * ((b - p.rho*p.xi*u*i - d)*T - 2.0*cmath.log((1 - g*exp_dT)/(1 - g))))
        D = ((b - p.rho*p.xi*u*i - d)/(p.xi**2)) * ((1 - exp_dT)/(1 - g*exp_dT))
        return cmath.exp(C + D * p.v0)

    # Heston European option price via integration (original 1993 formulation)
    def _Pj(self, j: int, lnK: float, T: float) -> float:
        # P1 and P2 integrals
        i = 1j
        p = self.p
        def integrand(u: float) -> float:
            u_c = complex(u, 0.0)
            if j == 1:
                phi = self.charfunc(u_c - i, T)
                denom = p.S0 * math.exp((p.r - p.q) * T)
            else:  # j == 2
                phi = self.charfunc(u_c, T)
                denom = 1.0
            val = cmath.exp(-i * u_c * lnK) * phi / (i * u_c)
            return val.real / denom
        integral = _safe_quad(integrand, 0.0, 200.0)  # upper limit 200 good in practice
        return 0.5 + (1.0 / math.pi) * integral

    def price_call_put(self, K: float, T: float, flag: Literal["call","put"]="call") -> float:
        if T <= 0:
            # intrinsic value under carries
            df_r = math.exp(-self.p.r * 0.0)
            df_q = math.exp(-self.p.q * 0.0)
            forward = self.p.S0 * df_q / df_r
            if flag == "call":
                return max(self.p.S0 - K, 0.0)
            else:
                return max(K - self.p.S0, 0.0)

        lnK = math.log(K)
        P1 = self._Pj(1, lnK, T)
        P2 = self._Pj(2, lnK, T)
        call = self.p.S0 * math.exp(-self.p.q * T) * P1 - K * math.exp(-self.p.r * T) * P2
        if flag == "call":
            return float(call)
        else:
            # Put via parity
            df_r = math.exp(-self.p.r * T)
            df_q = math.exp(-self.p.q * T)
            forward = self.p.S0 * df_q / df_r
            put = call - df_r * (forward - K)
            return float(put)

    # Implied vol for a single strike/maturity (call/put)
    def implied_vol(self, K: float, T: float, flag: Literal["call","put"]="call") -> float:
        price = self.price_call_put(K, T, flag)
        return bs_implied_vol(self.p.S0, K, T, self.p.r, self.p.q, price, flag)

    # Implied volatility smile for a vector of strikes (single T)
    def iv_smile(self, strikes: np.ndarray, T: float, flag: Literal["call","put"]="call") -> np.ndarray:
        ivs = []
        for K in strikes:
            ivs.append(self.implied_vol(float(K), T, flag))
        return np.array(ivs, dtype=float)

    # --------------------------
    # Simulation
    # --------------------------

    def simulate_paths(
        self,
        T: float,
        steps: int = 252,
        n_paths: int = 1000,
        scheme: Literal["QE","Euler"] = "QE",
        antithetic: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate S and v paths. Returns (tgrid, S_paths, v_paths)
        Shapes: S_paths, v_paths -> (n_paths, steps+1)
        """
        rng = np.random.default_rng(seed)
        dt = T / steps
        p = self.p

        n = n_paths
        if antithetic:
            n_half = (n + 1) // 2
        else:
            n_half = n

        S = np.full((n, steps+1), float(p.S0), dtype=float)
        v = np.full((n, steps+1), float(p.v0), dtype=float)

        # Correlated normals
        for t in range(steps):
            if antithetic:
                z1 = rng.standard_normal(n_half)
                z2 = rng.standard_normal(n_half)
                z1 = np.concatenate([z1, -z1])[:n]
                z2 = np.concatenate([z2, -z2])[:n]
            else:
                z1 = rng.standard_normal(n)
                z2 = rng.standard_normal(n)

            # Create correlated Brownian increments
            dW1 = np.sqrt(dt) * z1
            dW2 = np.sqrt(dt) * (p.rho * z1 + np.sqrt(1 - p.rho**2) * z2)

            if scheme == "Euler":
                vt = np.maximum(v[:, t], 0.0)
                v_next = vt + p.kappa * (p.theta - vt) * dt + p.xi * np.sqrt(np.maximum(vt, 0.0)) * dW2
                v_next = np.maximum(v_next, 0.0)
            else:
                # Andersen QE scheme
                vt = np.maximum(v[:, t], 0.0)
                m = p.theta + (vt - p.theta) * np.exp(-p.kappa * dt)
                s2 = (vt * p.xi**2 * np.exp(-p.kappa * dt) / p.kappa) * (1 - np.exp(-p.kappa * dt)) \
                     + (p.theta * p.xi**2 / (2 * p.kappa)) * (1 - np.exp(-p.kappa * dt))**2
                psi = s2 / (m*m)
                u = rng.random(n)
                v_next = np.empty_like(vt)
                # Two-regime QE
                psi_c = 1.5
                idx1 = psi <= psi_c
                idx2 = ~idx1
                # Regime 1: non-central chi-square approximation
                b2 = 2.0 / psi[idx1] - 1.0 + np.sqrt(2.0 / psi[idx1]) * np.sqrt(2.0 / psi[idx1] - 1.0)
                a = m[idx1] / (1.0 + b2)
                v_next[idx1] = a * (np.sqrt(b2) + np.tan(np.pi * (u[idx1] - 0.5)))**2
                # Regime 2: Exponential approximation
                p_exp = (psi[idx2] - 1.0) / (psi[idx2] + 1.0)
                beta = (1.0 - p_exp) / m[idx2]
                v_next[idx2] = np.where(u[idx2] > p_exp, -np.log((1.0 - u[idx2])/(1.0 - p_exp)) / beta, 0.0)
                # guard
                v_next = np.maximum(v_next, 0.0)

            # Asset update (log-Euler with drift adjustment)
            # Using discretization consistent with QE (Andersen): dS/S = (r - q - 0.5 v_t) dt + sqrt(v_t) dW1
            v_mid = 0.5 * (np.maximum(v[:, t], 0.0) + v_next)  # trapezoidal avg for stability
            S[:, t+1] = S[:, t] * np.exp((p.r - p.q - 0.5 * v_mid) * dt + np.sqrt(np.maximum(v_mid, 1e-14)) * dW1)
            v[:, t+1] = v_next

        tgrid = np.linspace(0.0, T, steps+1)
        return tgrid, S, v
