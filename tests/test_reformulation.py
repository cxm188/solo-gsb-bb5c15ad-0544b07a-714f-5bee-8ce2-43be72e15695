"""目标过敏原最小改配方案测试（不依赖 FastAPI，走 db + review + reformulation 全链路）。

覆盖：
- 复用审查记录当前配方树（两版取 B）列目标过敏原的全部来源路径；
- 树形动态规划最低总成本（对照枚举节点子集）与祖先阻断（不再选后代）；
- 同成本按节点路径序列字典序取最小、并列数量与 Fraction 精确并列；
- 无解时列出仍不可覆盖的来源；
- 缺失 / 需复核记录禁止使用；成本与路径校验；全程只读、不改写审查记录。
"""
import itertools
import os
import tempfile
import unittest
from fractions import Fraction

_DB_DIR = tempfile.mkdtemp(prefix="allergen-reform-")
os.environ["DB_PATH"] = os.path.join(_DB_DIR, "reform.db")

from app import db  # noqa: E402
from app.recheck import build_recheck  # noqa: E402
from app.reformulation import (  # noqa: E402
    ReformulationError, build_reformulation_preview, get_source_context,
    list_reformulation_records, solve_tree_dp)
from app.review import analyze  # noqa: E402

DICT_TEXT = """\
小麦粉：麸质谷物
乳清粉：乳
乳糖：乳
奶粉：
可可脂：
大豆卵磷脂：大豆
鸡蛋：蛋类
椰子粉：坚果
芝麻：芝麻
食盐：
"""
# 乳来源：巧克力酱 › 奶粉 › 乳清粉、巧克力酱 › 奶粉 › 乳糖（奶粉自身不标注）
TEXT_A = "配料表：小麦粉，巧克力酱（可可脂，奶粉（乳清粉，乳糖），大豆卵磷脂），鸡蛋，食盐"
# B 版无乳，坚果来源为 巧克力酱 › 椰子粉
TEXT_B = "配料表：小麦粉，巧克力酱（可可脂，椰子粉），鸡蛋，芝麻酱（芝麻），食盐"
# 父节点自身标注：奶粉自身即来源（不选奶粉时不能靠子树覆盖该来源）
TEXT_SELF = "巧克力酱（奶粉（乳清粉））"


def make_review(text_a, text_b=None, name="改配测试", with_fingerprint=True,
                dict_text=DICT_TEXT):
    db.bulk_import(dict_text)
    result, fp = analyze(text_a=text_a, text_b=text_b, label_a="现版配方",
                         label_b="新版配方", declared=[])
    return db.create_review(
        name=name, declared=[], text_a=text_a, text_b=text_b,
        label_a="现版配方", label_b="新版配方", result=result,
        dict_fingerprint=fp if with_fingerprint else None)


def paths_of(ctx):
    return {tuple(n["path"]) for n in ctx["candidate_nodes"]}


def brute_force(nodes, sources, price):
    """枚举节点子集：覆盖全部来源且互不为祖先/后代的最低成本集合。

    返回 (最小成本, 排序后的路径序列, 并列方案数)；无解返回 None。
    """
    best, best_key, ways = None, None, 0
    for k in range(len(nodes) + 1):
        for combo in itertools.combinations(nodes, k):
            chosen = set(combo)
            # 成本
            if any(p not in price for p in chosen):
                continue
            # 互不为祖先 / 后代
            ordered = sorted(chosen)
            if any(a != b and (b[:len(a)] == a or a[:len(b)] == b)
                   for a, b in itertools.combinations(ordered, 2)):
                continue
            # 覆盖全部来源
            def covered(sp):
                return any(sp[:len(p)] == p for p in chosen)
            if not all(covered(sp) for sp in sources):
                continue
            cost = sum((price[p] for p in chosen), Fraction(0))
            key = tuple(sorted(chosen))
            if best is None or cost < best:
                best, best_key, ways = cost, key, 1
            elif cost == best:
                ways += 1
                best_key = min(best_key, key)
    return None if best is None else (best, best_key, ways)


class TreeDPTest(unittest.TestCase):
    """纯计算：树形 DP、祖先阻断、并列与无解（不触碰数据库）。"""

    # 树根 A，子 B、C；B 子 D、E。来源：B›D、C
    NODES = [("A",), ("A", "B"), ("A", "B", "D"), ("A", "B", "E"), ("A", "C")]
    SOURCES = {("A", "B", "D"), ("A", "C")}

    def _run(self, price):
        price = {tuple(k.split("/")) if isinstance(k, str) else tuple(k):
                 Fraction(str(v)) for k, v in price.items()}
        return solve_tree_dp([tuple(p) for p in self.NODES], self.SOURCES, price)

    def test_pick_ancestor_blocks_subtree(self):
        # B 标价 1（阻断 D 子树），C 标价 1，A 更贵 → 选 B + C
        r = self._run({"A/B": 1, "A/C": 1, "A": 10, "A/B/D": 5})
        self.assertEqual(r[0], Fraction(2))
        self.assertEqual(r[1], (("A", "B"), ("A", "C")))
        self.assertEqual(r[2], 1)

    def test_ancestor_replaces_descendants(self):
        # A 标价 2，叶子合计 B(D)=1 + C=1 = 2 并列；字典序 (A,) 更小
        r = self._run({"A": 2, "A/B/D": 1, "A/C": 1})
        self.assertEqual(r[0], Fraction(2))
        self.assertEqual(r[1], (("A",),))
        self.assertEqual(r[2], 2)

    def test_self_source_cannot_skip(self):
        # B 自身也是来源时，只选叶子 D 无法覆盖 B；必须选 B 或 A
        sources = self.SOURCES | {("A", "B")}
        price = {("A", "B", "D"): Fraction(1), ("A", "C"): Fraction(1)}
        self.assertIsNone(solve_tree_dp([tuple(p) for p in self.NODES],
                                        sources, price))
        price[("A", "B")] = Fraction(1)
        r = solve_tree_dp([tuple(p) for p in self.NODES], sources, price)
        self.assertEqual(r[1], (("A", "B"), ("A", "C")))

    def test_infeasible_when_no_price_on_source_path(self):
        # C 是来源但整条路径无报价 → 无解
        r = self._run({"A/B/D": 1, "A/B": 1})
        self.assertIsNone(r)

    def test_matches_brute_force(self):
        price = {"A": 3, "A/B": 2, "A/B/D": 4, "A/B/E": 1, "A/C": 5}
        r = self._run(price)
        expect = brute_force([tuple(p) for p in self.NODES], self.SOURCES,
                             {tuple(k.split("/")): Fraction(str(v))
                              for k, v in price.items()})
        self.assertEqual((r[0], r[1], r[2]), expect)

    def test_matches_brute_force_random_like(self):
        # 每个节点独立报价，逐组对照暴力枚举
        nodes = [("R",), ("R", "X"), ("R", "X", "X1"), ("R", "X", "X2"),
                 ("R", "Y"), ("R", "Y", "Y1")]
        sources = {("R", "X", "X1"), ("R", "X", "X2"), ("R", "Y", "Y1")}
        for costs in itertools.product(range(1, 4), repeat=len(nodes)):
            price = {p: Fraction(c) for p, c in zip(nodes, costs)}
            r = solve_tree_dp([tuple(p) for p in nodes], sources, price)
            expect = brute_force([tuple(p) for p in nodes], sources, price)
            self.assertEqual(None if r is None else (r[0], r[1], r[2]),
                             expect, (costs,))

    def test_fraction_tie(self):
        # 0.1 + 0.2 与 0.3 二进制浮点不等，Fraction 精确化后应判并列
        r = self._run({"A": 0.3, "A/B/D": 0.1, "A/C": 0.2})
        self.assertEqual(float(r[0]), 0.3)
        self.assertEqual(r[1], (("A",),))
        self.assertEqual(r[2], 2)


class PreviewFlowTest(unittest.TestCase):
    """全链路：来源路径、两版取 B、候选节点、求解结果与只读性。"""

    def setUp(self):
        db.DB_PATH = os.path.join(
            _DB_DIR, f"reform_{self.id().rsplit('.', 1)[-1]}.db")
        db.init_db()

    def _fp(self, rid):
        with db._connect() as conn:
            return conn.execute(
                "SELECT dict_fingerprint FROM reviews WHERE id=?",
                (rid,)).fetchone()[0]

    def test_sources_and_candidate_nodes(self):
        rid = make_review(TEXT_A)
        ctx = get_source_context(rid, "乳")
        self.assertEqual(ctx["allergen"], "乳")
        self.assertEqual(ctx["review"]["source_version"], "现版配方")
        self.assertEqual(ctx["sources"], [
            ["巧克力酱", "奶粉", "乳清粉"],
            ["巧克力酱", "奶粉", "乳糖"],
        ])
        # 候选 = 来源路径前缀闭包；巧克力酱/奶粉自身不标注乳（has_own=False）
        by_path = {tuple(n["path"]): n["has_own"]
                   for n in ctx["candidate_nodes"]}
        self.assertEqual(set(by_path), {
            ("巧克力酱",), ("巧克力酱", "奶粉"),
            ("巧克力酱", "奶粉", "乳清粉"),
            ("巧克力酱", "奶粉", "乳糖"),
        })
        self.assertTrue(by_path[("巧克力酱", "奶粉", "乳清粉")])
        self.assertFalse(by_path[("巧克力酱",)])

    def test_two_versions_uses_b(self):
        rid = make_review(TEXT_A, TEXT_B, name="两版")
        recs = {r["id"]: r for r in list_reformulation_records()}
        self.assertEqual(recs[rid]["source_version"], "新版配方")
        # 两版取 B：B 中无乳 → 乳无来源报错；坚果来源为 巧克力酱 › 椰子粉
        with self.assertRaises(ReformulationError):
            get_source_context(rid, "乳")
        ctx = get_source_context(rid, "坚果")
        self.assertEqual(ctx["sources"], [["巧克力酱", "椰子粉"]])

    def test_optimal_and_blocked_sources(self):
        rid = make_review(TEXT_A)
        costs = [
            {"path": ["巧克力酱"], "cost": 10},
            {"path": ["巧克力酱", "奶粉"], "cost": 3},
            {"path": ["巧克力酱", "奶粉", "乳清粉"], "cost": 1},
            {"path": ["巧克力酱", "奶粉", "乳糖"], "cost": 1},
        ]
        # 叶子合计 2 < 奶粉 3 < 巧克力酱 10 → 选两个叶子（路径按字典序排列）
        r = build_reformulation_preview(rid, "乳", costs)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["total_cost"], 2)
        self.assertEqual(r["optimal_count"], 1)
        self.assertEqual([s["path"] for s in r["selected"]],
                         [["巧克力酱", "奶粉", "乳清粉"],
                          ["巧克力酱", "奶粉", "乳糖"]])
        for s in r["selected"]:
            self.assertEqual(len(s["blocked_sources"]), 1)
            self.assertEqual(s["blocked_sources"][0], s["path"])
        # 奶粉降到 2 → 与叶子合计并列：(奶粉) 路径序列字典序更小
        costs[1]["cost"] = 2
        r = build_reformulation_preview(rid, "乳", costs)
        self.assertEqual(r["total_cost"], 2)
        self.assertEqual(r["optimal_count"], 2)
        [sel] = r["selected"]
        self.assertEqual(sel["path"], ["巧克力酱", "奶粉"])
        self.assertEqual(sel["cost"], 2)
        self.assertEqual(sorted(sel["blocked_sources"]), [
            ["巧克力酱", "奶粉", "乳清粉"],
            ["巧克力酱", "奶粉", "乳糖"],
        ])
        # 所选节点互不为祖先 / 后代
        self.assertTrue(all(
            not (a != b and (b[:len(a)] == a or a[:len(b)] == b))
            for a, b in itertools.combinations(
                [tuple(s["path"]) for s in r["selected"]], 2)))

    def test_self_source_node(self):
        db.bulk_import("奶粉：乳\n乳清粉：乳\n")
        result, fp = analyze(text_a=TEXT_SELF, text_b=None,
                             label_a="现版配方", label_b="新版配方", declared=[])
        rid = db.create_review(name="自身标注", declared=[], text_a=TEXT_SELF,
                               text_b=None, label_a="现版配方",
                               label_b="新版配方", result=result,
                               dict_fingerprint=fp)
        # 只报价 乳清粉：乳清粉来源可由其自身覆盖，但 奶粉 自身来源不可覆盖
        # → 整体无解；不可覆盖清单只含奶粉自身来源
        r = build_reformulation_preview(
            rid, "乳", [{"path": ["巧克力酱", "奶粉", "乳清粉"], "cost": 1}])
        self.assertFalse(r["feasible"])
        self.assertEqual(r["uncoverable_sources"],
                         [["巧克力酱", "奶粉"]])
        # 报价奶粉后可行（一次阻断两条来源）
        r = build_reformulation_preview(rid, "乳", [
            {"path": ["巧克力酱", "奶粉"], "cost": 5}])
        self.assertTrue(r["feasible"])
        self.assertEqual(r["total_cost"], 5)
        [sel] = r["selected"]
        self.assertEqual(sel["blocked_sources"], [
            ["巧克力酱", "奶粉"], ["巧克力酱", "奶粉", "乳清粉"]])

    def test_all_unpriced_is_infeasible(self):
        rid = make_review(TEXT_A)
        r = build_reformulation_preview(rid, "乳", [])
        self.assertFalse(r["feasible"])
        self.assertEqual(r["total_cost"] is None, True)
        self.assertEqual(len(r["uncoverable_sources"]), 2)

    def test_unknown_allergen_rejected(self):
        rid = make_review(TEXT_A)
        with self.assertRaises(ReformulationError):
            get_source_context(rid, "鱼")
        with self.assertRaises(ReformulationError):
            build_reformulation_preview(rid, "鱼", [])

    def test_cost_validation(self):
        rid = make_review(TEXT_A)
        # 路径不在来源路径上
        with self.assertRaises(ReformulationError):
            build_reformulation_preview(
                rid, "乳", [{"path": ["小麦粉"], "cost": 1}])
        # 负数 / 非数字
        with self.assertRaises(ReformulationError):
            build_reformulation_preview(
                rid, "乳", [{"path": ["巧克力酱"], "cost": -1}])
        with self.assertRaises(ReformulationError):
            build_reformulation_preview(
                rid, "乳", [{"path": ["巧克力酱"], "cost": "abc"}])

    def test_missing_record_blocked(self):
        with self.assertRaises(ReformulationError) as ctx:
            get_source_context(999999, "乳")
        [p] = ctx.exception.problems
        self.assertEqual(p["reason"], "missing")
        self.assertIn("不存在", p["message"])

    def test_stale_record_blocked_until_rechecked(self):
        rid = make_review(TEXT_A)
        db.upsert_entry("乳清粉", ["乳", "花生"])  # 映射变更 → 需复核
        self.assertTrue(db.get_review(rid)["needs_recheck"])
        with self.assertRaises(ReformulationError) as ctx:
            get_source_context(rid, "乳")
        [p] = ctx.exception.problems
        self.assertEqual(p["reason"], "stale")
        self.assertIn("需复核", p["message"])
        recs = {r["id"]: r for r in list_reformulation_records()}
        self.assertFalse(recs[rid]["eligible"])
        self.assertIn("需复核", recs[rid]["blocked_reason"])
        # 复核后恢复
        outcome = build_recheck(db.get_review(rid))
        db.update_review_result(rid, outcome["new_result"],
                                outcome["fingerprint"])
        ctx2 = get_source_context(rid, "乳")
        self.assertEqual(len(ctx2["sources"]), 2)

    def test_legacy_record_blocked(self):
        rid = make_review(TEXT_A, with_fingerprint=False)
        with self.assertRaises(ReformulationError) as ctx:
            get_source_context(rid, "乳")
        self.assertEqual(ctx.exception.problems[0]["reason"], "legacy")

    def test_readonly(self):
        rid = make_review(TEXT_A)
        before = db.get_review(rid)
        costs = [{"path": n["path"], "cost": 1}
                 for n in get_source_context(rid, "乳")["candidate_nodes"]]
        build_reformulation_preview(rid, "乳", costs)
        get_source_context(rid, "乳")
        self.assertEqual(db.get_review(rid), before)
        self.assertEqual(self._fp(rid), before["dict_fingerprint"])
        self.assertEqual(len(db.list_reviews()), 1)


if __name__ == "__main__":
    unittest.main()
