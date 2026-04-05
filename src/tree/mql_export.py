from __future__ import annotations

from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

from src.tree.feature_matrix import DEFAULT_TREE_FEATURE_COLUMNS


def leaf_positive_probability(model: DecisionTreeClassifier, node: int) -> float:
    total = float(model.tree_.value[node][0].sum())
    if total <= 0:
        return 0.0
    class_lookup = {int(cls): i for i, cls in enumerate(model.classes_)}
    if 1 not in class_lookup:
        return 0.0
    return float(model.tree_.value[node][0][class_lookup[1]] / total)


def export_tree_to_mql5(
    model: DecisionTreeClassifier,
    output_path: str | Path,
    function_name: str = "EvaluateDecisionTreeProbability",
) -> Path:
    output_path = Path(output_path)
    feature_names = list(getattr(model, "feature_names_in_", DEFAULT_TREE_FEATURE_COLUMNS))
    tree = model.tree_

    def recurse(node: int, depth: int) -> list[str]:
        indent = "   " * depth
        if tree.feature[node] == -2:
            probability = leaf_positive_probability(model, node)
            return [f"{indent}return {probability:.10f};"]

        name = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        lines: list[str] = [f"{indent}if ({name} <= {threshold:.10f})", f"{indent}{{"]
        lines.extend(recurse(tree.children_left[node], depth + 1))
        lines.append(f"{indent}}}")
        lines.append(f"{indent}else")
        lines.append(f"{indent}{{")
        lines.extend(recurse(tree.children_right[node], depth + 1))
        lines.append(f"{indent}}}")
        return lines

    lines = [f"double {function_name}("]
    for idx, name in enumerate(feature_names):
        suffix = "," if idx < len(feature_names) - 1 else ""
        lines.append(f"   const double {name}{suffix}")
    lines.extend([")", "{"])
    lines.extend(recurse(0, 1))
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
