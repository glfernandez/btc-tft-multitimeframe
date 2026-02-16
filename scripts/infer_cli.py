"""Public CLI for individual/ensemble multi-timeframe TFT inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inference_pipeline import MultiTimeframePredictor
from futureview_adapter import ensemble_to_futureview
from mpc_adapter import extract_tvp_vector

VALID_TIMEFRAMES = ("15min", "4hr", "1day", "1week")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timestamp", required=True, help="Prediction timestamp, e.g. '2024-11-20 12:00:00'")
    p.add_argument("--mode", choices=["individual", "ensemble"], default="ensemble")
    p.add_argument("--timeframe", choices=list(VALID_TIMEFRAMES), help="Required for --mode individual")
    p.add_argument("--steps", choices=["1", "2"], default="1", help="Number of forecast steps")
    p.add_argument("--raw-data-path", type=Path, help="Override default raw 1-min CSV path")
    p.add_argument("--device", choices=["mps", "cuda", "cpu"], help="Inference device")
    p.add_argument("--include-futureview", action="store_true", help="Include FutureView adapted output (ensemble mode)")
    p.add_argument("--include-mpc", action="store_true", help="Include MPC tvp vector (ensemble mode)")
    p.add_argument("--output", type=Path, help="Optional JSON output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "individual" and not args.timeframe:
        raise SystemExit("--timeframe is required for --mode individual")

    predictor = MultiTimeframePredictor(raw_data_path=args.raw_data_path, device=args.device)
    predictions = predictor.predict_at_timestamp(args.timestamp, steps=args.steps)

    if args.mode == "individual":
        result = {args.timeframe: predictions.get(args.timeframe)}
    else:
        result = {"predictions": predictions}
        if args.include_futureview:
            first = predictions.get("15min")
            if first is not None:
                current_price = float(first["last_close"])
                result["futureview"] = ensemble_to_futureview(predictions, current_price, args.timestamp)
        if args.include_mpc:
            first = predictions.get("15min")
            if first is not None:
                current_price = float(first["last_close"])
                result["mpc_tvp_vector"] = extract_tvp_vector(predictions, current_price)

    payload = _to_jsonable(result)
    text = json.dumps(payload, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Saved inference output to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
