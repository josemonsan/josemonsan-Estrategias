"""
sp500_optimize.py
=======================================================================
Aplica BTC_Long_6H (strategies/BTC_Long_6H) a cada valor del S&P 500,
optimiza sus parámetros por valor y genera un script Pine por valor en
strategies/SP500_Long_D1/<TICKER>_Long_D1.pine.

Timeframe: DIARIO. Yahoo solo sirve intradía de los últimos ~730 días,
insuficiente para una SMA800 de 6H (~400 días de calentamiento) más un
backtest. Se usan las longitudes equivalentes en D1: SMA 200 D1 (= SMA800
6H en cripto, 4 velas/día), pendiente y ruptura de 5 días (= 20 velas 6H).
El motor es el mismo (_run de macd_sma800_6h_backtest.py): en un gráfico
diario, request.security("D", x[1], lookahead_on) es el valor del día
anterior, que es lo que replica features().

Protocolo por valor:
  1. rejilla (GRID) solo largos sobre TRAIN (2006-2018)
  2. selección por Calmar suavizado en meseta (media con vecinos) -> se
     mide fuera de muestra en TEST (2019-hoy)
  3. comparación OOS contra: config universal BTC_Long_6H, la mejor
     config única para todo el índice (elegida en TRAIN) y buy & hold
  4. los parámetros del .pine se eligen igual pero sobre todo el periodo
     (2006-hoy); el OOS del paso 2 es la estimación honesta de su valor

Uso:
    python3 sp500_optimize.py fetch        # constituyentes + precios D1 (data/sp500/)
    python3 sp500_optimize.py optimize     # rejilla, resultados y .pine
"""
from __future__ import annotations

import argparse
import io
import itertools
import multiprocessing as mp
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import macd_sma800_6h_backtest as m

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sp500"
OUT = ROOT / "strategies" / "SP500_Long_D1"
TEMPLATE = ROOT / "strategies" / "BTC_Long_6H" / "BTC_Long_6H.pine"

T_START, T_SPLIT = pd.Timestamp("2006-01-01"), pd.Timestamp("2019-01-01")
SLOPE_LEN = RE_LEN = 5            # 20 velas 6H = 5 días
MIN_TRADES = 5

GRID = dict(smaLen=[150, 200, 250], sl=[2.0, 3.0, 4.0], ch=[4.0, 6.0, 8.0],
            adxThr=[0.0, 15.0, 20.0, 25.0],   # 0 = filtro ADX desactivado
            slope=[False, True], hold=[False, True])
AXES = list(GRID)
ORDINAL = ["smaLen", "sl", "ch", "adxThr"]      # ejes suavizados en la meseta
UNIVERSAL = dict(smaLen=200, sl=3.0, ch=8.0, adxThr=20.0, slope=False, hold=False)  # BTC_Long_6H


# ----------------------------------------------------------------------
def fetch():
    from fetch_data import fetch_prices
    DATA.mkdir(parents=True, exist_ok=True)
    html = urllib.request.urlopen(urllib.request.Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode()
    t = pd.read_html(io.StringIO(html))[0]
    t = t.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    t[["ticker", "name", "sector"]].to_csv(DATA / "constituents.csv", index=False)
    px = fetch_prices([s.replace(".", "-") for s in t.ticker], start="2005-01-01")
    pd.to_pickle(px, DATA / "prices_d1.pkl")
    print(f"{len(px)}/{len(t)} valores descargados -> {DATA}")


# ----------------------------------------------------------------------
def metrics(eq, pnl_n, i0, i1, idx, cap=10000.0):
    e = eq[i0:i1] / eq[i0] * cap
    pk = np.maximum.accumulate(e)
    dd = ((pk - e) / pk).max() * 100
    net = (e[-1] / cap - 1) * 100
    yrs = max((idx[i1 - 1] - idx[i0]).days / 365.25, 1e-9)
    cagr = (max(1 + net / 100, 1e-9) ** (1 / yrs) - 1) * 100
    return net, cagr, dd, (cagr / dd if dd > 0 else 0.0), pnl_n


def run_ticker(df):
    """Rejilla completa para un valor -> arrays de métricas [TRAIN, TEST, FULL] con forma GRID."""
    idx = df.index
    i0 = int(np.searchsorted(idx, T_START))
    isp = int(np.searchsorted(idx, T_SPLIT))
    n = len(idx)
    shape = tuple(len(v) for v in GRID.values())
    res = {w: {k: np.full(shape, np.nan) for k in ("net", "cagr", "dd", "calmar", "n")} for w in ("train", "test", "full")}
    feats = {L: m.features(df, df, smaLen=L, slopeLen=SLOPE_LEN, reLen=RE_LEN) for L in GRID["smaLen"]}
    for pos in itertools.product(*[range(len(v)) for v in GRID.values()]):
        p = {k: GRID[k][j] for k, j in zip(AXES, pos)}
        F = feats[p["smaLen"]]
        for w, (a, b) in {"train": (i0, isp), "test": (isp, n), "full": (i0, n)}.items():
            if b - a < 250:
                continue
            s = lambda k: F[k][:b]
            eq, pnl, ent, ext, side = m._run(
                s("o"), s("h"), s("l"), s("c"), s("sma"), s("atrD"), s("adx"), s("macdD"), s("smaPrev"),
                s("hiRe"), s("loRe"), a, p["sl"], p["ch"], p["adxThr"] > 0, p["adxThr"], p["slope"], True,
                True, 1, p["hold"], 0.0035 / 100, 0.01, 10000.0)
            r = metrics(eq, len(pnl), a, b, idx)
            for k, v in zip(("net", "cagr", "dd", "calmar", "n"), r):
                res[w][k][pos] = v
    bh = {w: (df.close.iloc[b - 1] / df.close.iloc[a] - 1) * 100
          for w, (a, b) in {"train": (i0, isp), "test": (isp, n), "full": (i0, n)}.items()}
    return res, bh


def plateau(score, n_trades):
    """Media del Calmar con los vecinos ±1 en los ejes ordinales; penaliza pocos trades."""
    s = np.where(n_trades >= MIN_TRADES, score, np.nan)
    acc = np.nan_to_num(s).copy(); cnt = (~np.isnan(s)).astype(float)
    for ax in (AXES.index(a) for a in ORDINAL):
        for sh in (-1, 1):
            r = np.roll(s, sh, axis=ax)
            sl = [slice(None)] * s.ndim
            sl[ax] = 0 if sh == 1 else -1       # descartar el valor que da la vuelta
            r[tuple(sl)] = np.nan
            acc += np.nan_to_num(r); cnt += ~np.isnan(r)
    out = acc / np.maximum(cnt, 1)
    return np.where(np.isnan(s), -np.inf, out)


def pick(res_w):
    sm = plateau(res_w["calmar"], res_w["n"])
    pos = np.unravel_index(np.argmax(sm), sm.shape)
    return pos, {k: GRID[k][j] for k, j in zip(AXES, pos)}


def pos_of(p):
    return tuple(GRID[k].index(p[k]) for k in AXES)


# ----------------------------------------------------------------------
def pine_for(ticker_tv, p, stats):
    s = TEMPLATE.read_text(encoding="utf-8")
    name = f"{ticker_tv.replace('.', '_')}_Long_D1"

    def sub(pattern, repl):
        nonlocal s
        s2, k = re.subn(pattern, repl, s, count=1, flags=re.M)
        assert k == 1, pattern
        s = s2

    b = lambda x: "true" if x else "false"
    adx = f">{p['adxThr']:g}" if p["adxThr"] else "off"
    header = (
        f"// {name} — BTC_Long_6H (strategies/BTC_Long_6H) aplicada a {ticker_tv}\n"
        f"// en velas DIARIAS y optimizada para este valor con sp500_optimize.py.\n"
        f"// MACD D1 (régimen) + SMA {p['smaLen']} D1 (entrada) + Chandelier ATR D1 (salida), solo largos.\n"
        f"// Usar en gráfico diario (1D). Sin lookahead: D1 usa la vela diaria CERRADA anterior.\n"
        f"//\n"
        f"// Defaults: SMA {p['smaLen']} · Stop {p['sl']:g}×ATR · Chandelier {p['ch']:g}×ATR · ADX {adx} · "
        f"S {'on' if p['slope'] else 'off'} · H {'on' if p['hold'] else 'off'} · P off\n"
        f"// Réplica 2006-hoy: neto {stats['full_net']:.0f}% (B&H {stats['full_bh']:.0f}%) · "
        f"maxDD {stats['full_dd']:.0f}% · {stats['full_n']:.0f} ops\n"
        f"// Fuera de muestra 2019-hoy (params elegidos solo con 2006-2018): CAGR {stats['oos_cagr']:.1f}% "
        f"· maxDD {stats['oos_dd']:.0f}% · B&H CAGR {stats['bh_cagr']:.1f}%\n"
        f"// AVISO: optimización in-sample por valor sobre la lista ACTUAL del S&P 500 (sesgo de\n"
        f"// supervivencia). Ver strategies/SP500_Long_D1/README.md antes de operar.\n")
    s2, k = re.subn(r"^// BTC_Long_6H — .*?(?=^// =+\nstrategy)", lambda _: header, s, count=1, flags=re.M | re.S)
    assert k == 1
    s = s2
    sub(r'strategy\("BTC_Long_6H", shorttitle="BTC_Long_6H",', f'strategy("{name}", shorttitle="{name}",')
    sub(r'^smaLen  = input\.int\(\d+, "SMA \(velas del gráfico\)"',
        f'smaLen  = input.int({p["smaLen"]}, "SMA (velas del gráfico)"')
    sub(r'^useAdx  = input\.bool\(\w+,', f'useAdx  = input.bool({b(p["adxThr"] > 0)},')
    sub(r'^adxThr  = input\.float\([\d.]+,', f'adxThr  = input.float({p["adxThr"] or 20:g},')
    sub(r'^useSlope   = input\.bool\(\w+,', f'useSlope   = input.bool({b(p["slope"])},')
    sub(r'^slopeLen   = input\.int\(20, "Pendiente: velas atrás \(20 = 5 días en 6H\)"',
        f'slopeLen   = input.int({SLOPE_LEN}, "Pendiente: velas atrás (5 = 5 días en D1)"')
    sub(r'^reLen      = input\.int\(20,', f'reLen      = input.int({RE_LEN},')
    sub(r'^slMult     = input\.float\([\d.]+,', f'slMult     = input.float({p["sl"]:.1f},')
    sub(r'^chMult     = input\.float\([\d.]+,', f'chMult     = input.float({p["ch"]:.1f},')
    sub(r'^holdTrend    = input\.bool\(\w+,', f'holdTrend    = input.bool({b(p["hold"])},')
    sub(r'timeframe\.period == "360" \? "6H ✓" : "⚠ Diseñada para 6H"',
        'timeframe.isdaily ? "D ✓" : "⚠ Diseñada para D"')
    return name, s


# ----------------------------------------------------------------------
def optimize():
    px = pd.read_pickle(DATA / "prices_d1.pkl")
    cons = pd.read_csv(DATA / "constituents.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], []
    for tk_tv in cons.ticker:
        df = px.get(tk_tv.replace(".", "-"))
        # hace falta TRAIN con historia: cotizar desde antes de 2015
        if df is None or len(df) < 1000 or df.index[0] > pd.Timestamp("2015-01-01"):
            print(f"[skip] {tk_tv}: histórico insuficiente")
            continue
        jobs.append((tk_tv, df))
    with mp.Pool() as pool:
        out = pool.map(run_ticker, [df for _, df in jobs], chunksize=4)
    per = {tk: r for (tk, _), r in zip(jobs, out)}
    all_train_calmar = [np.where(res["train"]["n"] >= MIN_TRADES, res["train"]["calmar"], np.nan)
                        for res, _ in per.values()]
    # mejor configuración única para todo el índice, elegida solo con TRAIN
    med = np.nanmedian(np.stack(all_train_calmar), axis=0)
    upos = np.unravel_index(np.nanargmax(np.where(np.isnan(med), -np.inf, med)), med.shape)
    uni_best = {k: GRID[k][j] for k, j in zip(AXES, upos)}
    print("Mejor config única (TRAIN, mediana Calmar):", uni_best)
    for tk_tv, (res, bh) in per.items():
        cons_row = cons[cons.ticker == tk_tv].iloc[0]
        tr_pos, tr_p = pick(res["train"])
        fu_pos, fu_p = pick(res["full"])
        test_yrs = (pd.Timestamp.today() - T_SPLIT).days / 365.25
        bh_cagr = ((1 + bh["test"] / 100) ** (1 / test_yrs) - 1) * 100
        g = lambda w, k, pos: float(res[w][k][pos])
        row = dict(ticker=tk_tv, name=cons_row["name"], sector=cons_row.sector,
                   **{f"p_{k}": v for k, v in fu_p.items()},
                   full_net=g("full", "net", fu_pos), full_dd=g("full", "dd", fu_pos), full_n=g("full", "n", fu_pos),
                   full_bh=bh["full"],
                   oos_cagr=g("test", "cagr", tr_pos), oos_dd=g("test", "dd", tr_pos), oos_n=g("test", "n", tr_pos),
                   oos_uni_cagr=g("test", "cagr", pos_of(UNIVERSAL)), oos_uni_dd=g("test", "dd", pos_of(UNIVERSAL)),
                   oos_unibest_cagr=g("test", "cagr", upos), oos_unibest_dd=g("test", "dd", upos),
                   bh_cagr=bh_cagr, train_pick=str(tr_p))
        row["robust"] = bool(row["oos_cagr"] > 0 and row["oos_cagr"] >= row["oos_uni_cagr"] and row["full_n"] >= 10)
        name, src = pine_for(tk_tv, fu_p, row)
        (OUT / f"{name}.pine").write_text(src, encoding="utf-8")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUT / "optimization_results.csv", index=False)
    summary(df, uni_best)
    write_readme(df, uni_best)


def summary(df, uni_best):
    print(f"\n{len(df)} valores optimizados -> {OUT}")
    cols = {"Optimizada por valor": ("oos_cagr", "oos_dd"),
            "Universal BTC_Long_6H": ("oos_uni_cagr", "oos_uni_dd"),
            f"Mejor única TRAIN": ("oos_unibest_cagr", "oos_unibest_dd")}
    print("OOS 2019-hoy (mediana entre valores):")
    for k, (c, d) in cols.items():
        print(f"  {k:24s} CAGR {df[c].median():5.1f}%  maxDD {df[d].median():5.1f}%  "
              f"Calmar {np.nanmedian(df[c] / df[d]):.2f}  % valores CAGR>0: {(df[c] > 0).mean():.0%}")
    print(f"  {'Buy & hold':24s} CAGR {df.bh_cagr.median():5.1f}%")
    print(f"  Optimizada > universal en {(df.oos_cagr > df.oos_uni_cagr).mean():.0%} de valores; "
          f"> B&H en {(df.oos_cagr > df.bh_cagr).mean():.0%}")


def write_readme(df, uni_best):
    fmt = lambda p: (f"SMA {p['smaLen']} · Stop {p['sl']:g} · Ch {p['ch']:g} · "
                     f"ADX {'>' + format(p['adxThr'], 'g') if p['adxThr'] else 'off'} · "
                     f"S {'on' if p['slope'] else 'off'} · H {'on' if p['hold'] else 'off'}")
    rows = [("Optimizada por valor (elegida en 2006-18)", "oos_cagr", "oos_dd"),
            ("Universal BTC_Long_6H (SMA 200 · Stop 3 · Ch 8 · ADX>20)", "oos_uni_cagr", "oos_uni_dd"),
            (f"Mejor config única del índice en 2006-18 ({fmt(uni_best)})", "oos_unibest_cagr", "oos_unibest_dd")]
    t = "\n".join(f"| {n} | {df[c].median():.1f} % | {df[d].median():.1f} % | {np.nanmedian(df[c] / df[d]):.2f} | "
                  f"{(df[c] > 0).mean():.0%} | {(df[c] > df.bh_cagr).mean():.0%} |" for n, c, d in rows)
    t += f"\n| Buy & hold | {df.bh_cagr.median():.1f} % | — | — | {(df.bh_cagr > 0).mean():.0%} | — |"
    top = df[df.robust].sort_values("oos_cagr", ascending=False).head(25)
    tt = "\n".join(f"| {r.ticker} | {r.sector} | {r.oos_cagr:.1f} % | {r.oos_dd:.0f} % | {r.bh_cagr:.1f} % | "
                   f"{fmt({k[2:]: getattr(r, k) for k in df.columns if k.startswith('p_')})} |"
                   for r in top.itertuples())
    pc = {k: df[f"p_{k}"].value_counts(normalize=True).sort_index() for k in AXES}
    dist = "\n".join(f"| {k} | " + " · ".join(f"{v:g}: {w:.0%}" if not isinstance(v, (bool, np.bool_))
                                               else f"{'on' if v else 'off'}: {w:.0%}" for v, w in pc[k].items()) + " |"
                      for k in AXES)
    md = f"""# SP500_Long_D1

Estrategia **BTC_Long_6H** (`strategies/BTC_Long_6H`) aplicada a cada valor del S&P 500 y optimizada
por valor con `sp500_optimize.py`. Hay un script Pine por valor: `<TICKER>_Long_D1.pine` (los tickers
con punto, como BRK.B, usan guion bajo: `BRK_B_Long_D1.pine`). En total son {len(df)} valores; los que
empezaron a cotizar después de 2015 se excluyen porque no tienen periodo de entrenamiento.

## Adaptación a acciones

- **Timeframe diario (1D), no 6H.** Yahoo solo sirve datos intradía de los últimos ~730 días, que no
  alcanzan para calentar una SMA800 de 6H (~400 días) y hacer un backtest. Se usan longitudes
  equivalentes: SMA 200 D1 (= SMA800 6H en cripto), pendiente de 5 días y ruptura de 5 días (= 20 velas
  6H). La lógica Pine es idéntica; solo cambian los defaults.
- Son solo largos, igual que BTC_Long_6H. Comisión de 0,0035 %, 1 tick de deslizamiento y 100 % del
  equity por operación.

## Optimización

- **Rejilla por valor** ({int(np.prod([len(v) for v in GRID.values()]))} combinaciones):
  - SMA {GRID['smaLen']};
  - Stop {GRID['sl']} × ATR;
  - Chandelier {GRID['ch']} × ATR;
  - ADX off/15/20/25;
  - S on/off;
  - H on/off.
- **Selección:** Calmar (CAGR / maxDD) suavizado con los vecinos de la rejilla (meseta, no pico), con un
  mínimo de {MIN_TRADES} operaciones.
- **Test honesto:** los parámetros se eligen solo con 2006–2018 y se miden en 2019–hoy (fuera de muestra).
- **Parámetros del `.pine`:** se eligen igual pero con todo 2006–hoy. Su rendimiento futuro esperado es el
  fuera de muestra de abajo, no el in-sample de la cabecera de cada script.

## Resultado fuera de muestra (2019 → hoy, mediana entre valores)

| Configuración | CAGR | maxDD | Calmar | Valores con CAGR > 0 | Valores que baten B&H |
|---|---|---|---|---|---|
{t}

Optimizar por valor bate a la configuración universal en el {(df.oos_cagr > df.oos_uni_cagr).mean():.0%} de
los valores. Todo el detalle está en `optimization_results.csv`: parámetros elegidos, in-sample 2006–hoy,
fuera de muestra y B&H.

## Parámetros elegidos (distribución entre valores)

| Parámetro | Reparto |
|---|---|
{dist}

## Top 25 fuera de muestra (con la columna `robust` del CSV)

`robust` = CAGR fuera de muestra > 0, ≥ el de la configuración universal y ≥ 10 operaciones en 2006–hoy.

| Ticker | Sector | CAGR OOS | maxDD OOS | CAGR B&H | Params del .pine (2006–hoy) |
|---|---|---|---|---|---|
{tt}

## Avisos

- **Sesgo de supervivencia:** se usa la lista actual del S&P 500. Los valores que salieron del índice (los
  perdedores) no están, así que todos los resultados, B&H incluido, salen inflados.
- **Sobreajuste:** hay {int(np.prod([len(v) for v in GRID.values()]))} combinaciones por valor y
  ~{df.full_n.median():.0f} operaciones por valor en 20 años. El dato que importa es la comparación fuera
  de muestra entre optimizar por valor y usar una configuración única.
- Estrategia 100 % invertida en un solo valor: no es una cartera. Los drawdowns individuales son altos.
- Los scripts no se han ejecutado uno a uno en TradingView. Los números vienen de la réplica Python
  (validada contra TradingView en BTC).
"""
    (OUT / "README.md").write_text(md, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "optimize"])
    a = ap.parse_args()
    fetch() if a.cmd == "fetch" else optimize()


if __name__ == "__main__":
    sys.exit(main())
