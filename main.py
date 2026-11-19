#!/usr/bin/env python3
import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z_VALS = [-1.645, -0.674, 0.0, 0.674, 1.645]
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]

def round4(val):
    if not isinstance(val, (int, float, Decimal)):
        val = float(val)
    if math.isnan(val):
        val = 0.01
    d = Decimal(str(val))
    return float(d.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

def parse_time(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    except Exception:
        dt = EPOCH
    return (dt - EPOCH).total_seconds() / 3600.0

def clamp(val, low=0.01, high=100.0):
    return max(low, min(high, val))

def logit_clamp(val):
    return max(0.01, min(99.99, val))

def logit(pct):
    p = logit_clamp(pct) / 100.0
    return math.log(p / (1.0 - p))

def inv_logit(lg):
    lg = max(-20.0, min(20.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))

def least_squares(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, ys[0]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if abs(den) < 1e-12:
        return 0.0, mean_y
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept

def process_case(case_path):
    data = json.loads(Path(case_path).read_text(encoding='utf-8'))
    target_time = parse_time(data.get('target', '2026-11-26T00:00:00Z'))
    horizon_minutes = float(data.get('horizon_minutes', 1440))
    days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    trophies = data.get('trophies', [])
    num_trophies = len(trophies)
    history = data.get('history', [])

    # 1. Parse history and deduplicate distinct states
    states = []
    seen_state = None
    for entry in history:
        t = parse_time(entry.get('at', '2026-11-19T00:00:00Z'))
        players = float(entry.get('players_tracked', 1))
        rates = entry.get('rates', [])
        sig = (players, tuple(rates if rates else []))
        if sig != seen_state:
            states.append({
                'time': t,
                'players': players,
                'rates': rates
            })
            seen_state = sig

    last_read = history[-1] if history else {}
    last_players = float(last_read.get('players_tracked', 638))
    last_rates = last_read.get('rates', [None] * num_trophies)

    is_flat = len(states) <= 1 or all(s['players'] == states[0]['players'] for s in states)

    max_time = states[-1]['time'] if states else 0.0
    recent_states = [s for s in states if s['time'] >= max_time - 24.0]
    if not recent_states:
        recent_states = states

    # 2. Predict N(t) at target
    if is_flat or not recent_states:
        n_pred = last_players * 6.0
        n_is_flat_prior = True
    else:
        xs = [math.log(1.0 + s['time']) for s in recent_states]
        ys = [math.log(s['players']) for s in recent_states]
        slope_n, intercept_n = least_squares(xs, ys)
        
        target_x = math.log(1.0 + target_time)
        obs_max_x = xs[-1]
        extra_x = target_x - obs_max_x
        if extra_x > 0:
            adjusted_target_x = obs_max_x + 0.7 * extra_x
        else:
            adjusted_target_x = target_x

        log_n_pred = intercept_n + slope_n * adjusted_target_x
        n_pred = math.exp(log_n_pred)
        n_pred = max(n_pred, last_players)

        if len(states) >= 2:
            max_daily_g = 1.0
            for i in range(1, len(states)):
                dt = states[i]['time'] - states[i-1]['time']
                if dt > 0:
                    g = (states[i]['players'] / states[i-1]['players']) ** (24.0 / dt)
                    if g > max_daily_g:
                        max_daily_g = g
            max_possible_n = last_players * (max_daily_g ** (horizon_hours / 24.0))
            n_pred = min(n_pred, max(max_possible_n, last_players * 1.05))

        n_is_flat_prior = False

    # 3. Recover earned counts E_i(t) and predict E_i at target
    trophy_results = []
    
    for i, troph in enumerate(trophies):
        t_id = troph.get('id', i + 1)
        
        e_points = []
        for s in states:
            rates = s['rates']
            if rates and i < len(rates) and rates[i] is not None:
                r = float(rates[i])
                e = round(r * s['players'] / 100.0)
                e_points.append({'time': s['time'], 'E': e, 'rate': r})

        last_r = last_rates[i] if (last_rates and i < len(last_rates) and last_rates[i] is not None) else None
        if last_r is None:
            for s in reversed(states):
                if s['rates'] and i < len(s['rates']) and s['rates'][i] is not None:
                    last_r = float(s['rates'][i])
                    break
            if last_r is None:
                last_r = 0.5

        last_e = round(last_r * last_players / 100.0)
        is_100 = (last_r >= 99.95)

        valid_e_states = [ep for ep in e_points if ep['time'] >= max_time - 24.0]
        if not valid_e_states:
            valid_e_states = e_points

        residual = 0.15
        if len(valid_e_states) >= 3 and not is_100:
            xs_e = [math.log(1.0 + ep['time']) for ep in valid_e_states]
            ys_e = [math.log(max(0.0, ep['E']) + 0.5) for ep in valid_e_states]
            slope_e, intercept_e = least_squares(xs_e, ys_e)
            slope_e = max(0.0, min(2.0, slope_e))
            
            target_x = math.log(1.0 + target_time)
            log_e_pred = intercept_e + slope_e * target_x
            e_pred = math.exp(log_e_pred) - 0.5
            e_pred = max(e_pred, float(last_e))
            e_pred = min(e_pred, 0.99 * n_pred)

            resids = []
            for ep in valid_e_states:
                pred_val = math.exp(intercept_e + slope_e * math.log(1.0 + ep['time'])) - 0.5
                actual_val = max(0.0, ep['E'])
                resids.append((math.log(max(0.1, actual_val + 0.5)) - math.log(max(0.1, pred_val + 0.5))) ** 2)
            if resids:
                residual = max(0.15, math.sqrt(sum(resids) / len(resids)))
        else:
            e_pred = float(last_e)
            if n_is_flat_prior and is_100:
                e_pred = n_pred

        # 4. Median rate calculation
        if is_100:
            median_rate = 100.0
        elif n_is_flat_prior:
            median_rate = clamp(last_r * last_players / n_pred)
        else:
            median_rate = clamp(100.0 * e_pred / n_pred)

        # 5. Spread (logit sigma)
        staleness = 0.35
        h_hours = horizon_hours
        term_horizon = 0.35 * math.log(1.0 + h_hours / 24.0)
        term_rw = 0.5 * math.sqrt(days)
        
        sigma = math.sqrt((term_rw ** 2) + (residual ** 2) + (term_horizon ** 2) + (staleness ** 2))
        if is_flat:
            sigma = max(1.0, sigma)

        if is_100:
            q_vals = [99.5, 99.9, 100.0, 100.0, 100.0]
        else:
            lg_med = logit(median_rate)
            q_raw = [inv_logit(lg_med + z * sigma) for z in Z_VALS]
            q_clamped = [clamp(q) for q in q_raw]
            q_vals = []
            prev = 0.01
            for q in q_clamped:
                val = max(prev, q)
                q_vals.append(round4(val))
                prev = val
            for j in range(1, len(q_vals)):
                if q_vals[j] < q_vals[j-1]:
                    q_vals[j] = q_vals[j-1]

        trophy_results.append({
            'id': t_id,
            'q': [round4(x) for x in q_vals]
        })

    return {'trophies': trophy_results}

def main():
    input_dir = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
    output_dir = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    output_dir.mkdir(parents=True, exist_ok=True)

    in_files = sorted(input_dir.glob('*.in'))
    if not in_files:
        in_files = sorted(input_dir.glob('**/*.in'))

    if not in_files:
        print(f"No input files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    for inf in in_files:
        out_name = inf.stem + '.out'
        out_path = output_dir / out_name
        result = process_case(inf)
        out_path.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')

if __name__ == '__main__':
    main()
