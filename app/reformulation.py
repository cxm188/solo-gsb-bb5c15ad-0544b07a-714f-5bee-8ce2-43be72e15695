"""目标过敏原最小改配方案（临时预览，不改写审查记录）。

从**历史审查记录**中选择一条**时效正常**的记录与一个目标过敏原，复用其
**当前配方树**（两版记录取标签核对对象，即新版配方 B；单版取现版），列出
该过敏原在树中的**全部来源路径**（配料自身标注该过敏原的节点，传播路径与
正式审查的遗漏来源路径口径一致）。

用户按**完整节点路径**为可替换节点填写改配成本：

- 未填写成本的节点视为**不可替换**（不可选）；
- 选择某个祖先节点即同时阻断其子树内的全部来源，此时**不得再选其后代**；
- 后端用**树形动态规划**求覆盖全部来源路径的**最低总成本**节点集合：
  每个节点只有「选自己（覆盖整棵子树）」与「不选（各子树分别求解）」两种
  状态，自底向上一次遍历即可，不枚举节点子集组合；
- 总成本并列时按**所选节点路径序列**的字典序取最小者，保证结果唯一确定；
  同时统计达到最低成本的不同方案总数，供页面提示；
- 成本经 ``Fraction(str(...))`` 精确化，比较与并列判定不受浮点误差影响。

无解（存在既不可替换、也无任何可替换祖先的来源）时，列出仍不可覆盖的来源
路径。记录缺失（不存在 / 已删除）或处于**需复核**状态（旧记录无指纹、相关
原料映射已变更）时禁止使用，并指出原因。整个模块只读：不写入字典，也不改
写任何审查记录。
"""
from __future__ import annotations

import math
from fractions import Fraction
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

from . import db
from .analysis import missing_allergen_sources
from .changeover import blocked_message, product_of

# 节点路径（名称元组）→ 节点状态
Path = Tuple[str, ...]


class ReformulationError(ValueError):
    """改配方案请求无法完成（映射为 HTTP 400）。

    problems 逐条列出禁止使用的记录及原因，供页面定位。
    """

    def __init__(self, message: str, problems: Optional[List[dict]] = None):
        super().__init__(message)
        self.problems = problems or []


def list_reformulation_records() -> List[dict]:
    """候选记录清单：每条历史记录当前配方的过敏原集合与参与资格。"""
    records = []
    for summary in db.list_reviews():
        review = db.get_review(summary["id"])
        needs = bool(review.get("needs_recheck"))
        reason = review.get("recheck_reason")
        records.append({
            **product_of(review),
            "created_at": review["created_at"],
            "has_two_versions": bool(review.get("text_b")),
            "eligible": not needs,
            "blocked_reason": blocked_message(reason) if needs else None,
        })
    records.sort(key=lambda r: r["id"])
    return records


def _current_tree(review: dict):
    """记录当前配方树（两版取 B / 新版，单版取 A）与其版本标签。"""
    result = review.get("result") or {}
    label_b = review.get("label_b")
    if result.get("version_b") and result.get("label_check_version") == label_b:
        return result["version_b"]["tree"], result.get("label_check_version")
    version = result.get("version_b") or result.get("version_a") or {}
    return version.get("tree", []), result.get("label_check_version")


def _eligible_review(review_id: int):
    """取出记录并校验时效资格；缺失 / 需复核时抛出 ReformulationError。"""
    problems: List[dict] = []
    try:
        rid = int(review_id)
    except (TypeError, ValueError):
        raise ReformulationError(
            "记录 ID 无效，禁止使用",
            problems=[{"id": review_id, "name": None, "reason": "invalid",
                      "message": "记录 ID 无效，禁止使用"}])
    review = db.get_review(rid)
    if review is None:
        raise ReformulationError(
            "审查记录不存在或已删除，禁止使用",
            problems=[{"id": rid, "name": None, "reason": "missing",
                      "message": "审查记录不存在或已删除，禁止使用"}])
    if review.get("needs_recheck"):
        reason = review.get("recheck_reason")
        problems.append({
            "id": review["id"], "name": review["name"],
            "reason": reason or "unknown",
            "message": f"记录{blocked_message(reason)}，禁止使用",
        })
        raise ReformulationError(
            "所选记录处于需复核状态，请先完成字典时效复核", problems=problems)
    return review


def _allergen_sources(tree: List[dict], allergen: str) -> Set[Path]:
    """目标过敏原在当前配方树中的全部来源路径（自身标注节点）。"""
    mapped = missing_allergen_sources(tree, [allergen])
    return {tuple(p) for p in mapped.get(allergen, [])}


def _candidate_nodes(tree: List[dict], allergen: str,
                     sources: Set[Path]):
    """收集候选节点：子树内存在目标过敏原来源的全部节点（来源路径前缀闭包）。

    返回 (按先序遍历去重后的路径列表, {路径: 节点自身是否为来源})。
    同名原料出现在不同分支时由完整路径区分；同父下同名兄弟在路径结构上
    不可区分，按平台既有「路径结构去重」口径合并为同一候选节点。
    """
    nodes: List[Path] = []
    has_own: Dict[Path, bool] = {}
    seen: Set[Path] = set()

    def walk(items: List[dict], prefix: Path) -> None:
        for node in items:
            path = prefix + (node.get("name", ""),)
            own = allergen in set(node.get("own_allergens", ()))
            total = allergen in set(node.get("total_allergens", ()))
            if total or own:
                if path not in seen:
                    seen.add(path)
                    nodes.append(path)
                has_own[path] = has_own.get(path, False) or own
            walk(node.get("children", []), path)

    walk(tree, ())
    # 防御性校验：候选集合应恰好是来源路径的前缀闭包
    assert set(nodes) == {p[:i] for p in sources for i in range(1, len(p) + 1)}
    return nodes, has_own


def get_source_context(review_id: int, allergen: str) -> dict:
    """目标过敏原的来源路径与可填报成本的候选节点清单（只读）。"""
    review = _eligible_review(review_id)
    allergen = (allergen or "").strip()
    if not allergen:
        raise ReformulationError("目标过敏原不能为空")
    tree, label = _current_tree(review)
    sources = _allergen_sources(tree, allergen)
    if not sources:
        raise ReformulationError(
            f"记录「{review['name']}」的当前配方树中不存在过敏原「{allergen}」"
            "的任何来源，请改选其它过敏原")
    nodes, has_own = _candidate_nodes(tree, allergen, sources)
    return {
        "review": {"id": review["id"], "name": review["name"],
                   "source_version": label},
        "allergen": allergen,
        "sources": [list(p) for p in sorted(sources)],
        "candidate_nodes": [{"path": list(p), "has_own": has_own[p]}
                            for p in nodes],
    }


def _json_number(value: Fraction):
    """Fraction → JSON 可序列化数值（整数保留 int，其余转 float）。"""
    return int(value) if value.denominator == 1 else float(value)


def _is_ancestor(ancestor: Path, path: Path) -> bool:
    return len(ancestor) <= len(path) and path[:len(ancestor)] == ancestor


def solve_tree_dp(nodes: List[Path], sources: Set[Path],
                  price: Dict[Path, Fraction]):
    """树形动态规划：求覆盖全部来源路径的最低总成本节点集合。

    每个节点两种选择：
    - **选自己**（仅当填报了成本）：支付该节点成本，子树内全部来源一并阻断，
      后代不再参与选择；
    - **不选**：节点自身不能是来源，且各子树分别取最优后合并。

    状态值为 ``(成本, 所选节点路径序列, 方案数)``；成本并列时按路径序列
    字典序取最小，方案数相乘 / 相加统计并列总数。不可行为 ``None``。
    返回 (总成本, 排序后的所选路径元组, 并列方案数)，无解返回 ``None``。
    """
    order = list(nodes)                       # 先序（父先于子）
    children: Dict[Path, List[Path]] = {p: [] for p in order}
    tops: List[Path] = []
    for p in order:
        if len(p) == 1:
            tops.append(p)
        else:
            children[p[:-1]].append(p)

    state: Dict[Path, Optional[tuple]] = {}
    for p in reversed(order):                 # 后序自底向上
        options = []
        if p in price:
            # 选祖先：整棵子树一次阻断，方案键就是该节点自身路径
            options.append((price[p], (p,), 1))
        subs = [state[c] for c in children[p]]
        if p not in sources and all(s is not None for s in subs):
            # 不选自己：自身不是来源时才可由各子树分别覆盖
            if subs:
                cost = sum((s[0] for s in subs), Fraction(0))
                key = tuple(sorted(chain.from_iterable(s[1] for s in subs)))
                ways = math.prod(s[2] for s in subs)
            else:
                cost, key, ways = Fraction(0), (), 1
            options.append((cost, key, ways))
        if not options:
            state[p] = None
            continue
        best_cost = min(o[0] for o in options)
        # 成本并列时保留路径序列字典序最小者；并列数统计达到最低成本的
        # 全部不同选择（含键不同的「选自己 / 不选」并列），与换线排程、
        # 模板分组两个模块的并列口径一致。
        best_key = min(o[1] for o in options if o[0] == best_cost)
        tie_ways = sum(o[2] for o in options if o[0] == best_cost)
        state[p] = (best_cost, best_key, tie_ways)

    root_subs = [state[p] for p in tops]
    if not tops or any(s is None for s in root_subs):
        return None
    total = sum((s[0] for s in root_subs), Fraction(0))
    key = tuple(sorted(chain.from_iterable(s[1] for s in root_subs)))
    ways = math.prod(s[2] for s in root_subs)
    return total, key, ways


def _parse_cost(value, path: Path) -> Fraction:
    """成本校验：必须是有限且不小于 0 的数值（经 Fraction 精确化）。"""
    shown = " › ".join(path)
    if isinstance(value, bool):
        raise ReformulationError(f"节点「{shown}」的改配成本不是有效数字")
    try:
        cost = Fraction(str(value).strip())
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise ReformulationError(
            f"节点「{shown}」的改配成本不是有效数字：{value!r}")
    # NaN / ±Infinity 经 Fraction(str(...)) 会抛错，这里再兜底拒绝非有限值
    if not math.isfinite(float(cost)) or cost < 0:
        raise ReformulationError(
            f"节点「{shown}」的改配成本必须是不小于 0 的有限数值")
    return cost


def build_reformulation_preview(review_id: int, allergen: str,
                                cost_items: List[dict]) -> dict:
    """改配方案预览的完整编排：校验资格与成本后做树形 DP（只读）。

    cost_items 为 ``[{"path": [节点名, ...], "cost": 数值}, ...]``，
    未出现在列表中的候选节点视为不可替换。记录缺失 / 需复核、过敏原无
    来源、成本非法或路径不存在时抛出 :class:`ReformulationError`。
    """
    review = _eligible_review(review_id)
    allergen = (allergen or "").strip()
    if not allergen:
        raise ReformulationError("目标过敏原不能为空")
    tree, label = _current_tree(review)
    sources = _allergen_sources(tree, allergen)
    if not sources:
        raise ReformulationError(
            f"记录「{review['name']}」的当前配方树中不存在过敏原「{allergen}」"
            "的任何来源，请改选其它过敏原")
    nodes, _has_own = _candidate_nodes(tree, allergen, sources)
    node_set = set(nodes)

    price: Dict[Path, Fraction] = {}
    for item in cost_items or []:
        raw_path = item.get("path")
        if not isinstance(raw_path, list) or not raw_path or \
                not all(isinstance(x, str) and x.strip() for x in raw_path):
            raise ReformulationError("存在路径为空或格式无效的成本条目")
        path = tuple(x.strip() for x in raw_path)
        if path not in node_set:
            raise ReformulationError(
                f"成本节点「{' › '.join(path)}」不在过敏原「{allergen}」"
                "的来源路径上，请按完整节点路径重新填写")
        cost = _parse_cost(item.get("cost"), path)
        if path in price and price[path] != cost:
            raise ReformulationError(
                f"节点「{' › '.join(path)}」重复填写了不同成本，请只保留一条")
        price[path] = cost

    sorted_sources = sorted(sources)
    base = {
        "review": {"id": review["id"], "name": review["name"],
                   "source_version": label},
        "allergen": allergen,
        "sources": [list(p) for p in sorted_sources],
        "priced_nodes": [list(p) for p in sorted(price)],
    }
    solution = solve_tree_dp(nodes, sources, price)
    if solution is None:
        # 不可覆盖来源：路径上（含自身）没有任何可替换节点
        uncoverable = [
            list(p) for p in sorted_sources
            if not any(seg in price for seg in
                       (p[:i] for i in range(1, len(p) + 1)))]
        return {**base, "feasible": False, "total_cost": None,
                "optimal_count": 0, "selected": [],
                "uncoverable_sources": uncoverable}

    total, selected, optimal_count = solution
    blocked: Dict[Path, List[Path]] = {p: [] for p in selected}
    for sp in sorted_sources:
        owner = next(p for p in selected if _is_ancestor(p, sp))
        blocked[owner].append(sp)
    return {
        **base, "feasible": True,
        "total_cost": _json_number(total),
        "optimal_count": optimal_count,
        "selected": [{
            "path": list(p),
            "cost": _json_number(price[p]),
            "blocked_sources": [list(s) for s in blocked[p]],
        } for p in selected],
        "uncoverable_sources": [],
    }
