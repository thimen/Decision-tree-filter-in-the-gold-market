from __future__ import annotations

import json
from typing import Any

from src.backtest import simulate_with_trace
from src.config import RunSpec
from src.run.cli import parse_args
from src.run.configuration import build_run_spec, write_config_template
from src.run.pipeline import execute_experiment
from src.strategy import add_multi_indicator_columns, combine_vote, component_signals
from src.time_utils import parse_utc


def run_experiment(spec: RunSpec, *, mt5_module: Any | None = None) -> dict[str, str]:
    return execute_experiment(
        spec.experiment,
        spec.results_dir,
        mt5_module=mt5_module,
        parse_utc_fn=parse_utc,
        add_multi_indicator_columns_fn=add_multi_indicator_columns,
        component_signals_fn=component_signals,
        combine_vote_fn=combine_vote,
        simulate_with_trace_fn=simulate_with_trace,
    )


def main() -> None:
    args = parse_args()
    if args.write_config_template is not None:
        config_path = write_config_template(args.write_config_template)
        print(json.dumps({"config_template_path": str(config_path)}, indent=2))
        return

    result = run_experiment(build_run_spec(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
