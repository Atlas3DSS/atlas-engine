"""Ours, not upstream: Deterministic 4-Bank Mixture of Engrams (MoE-Engram) with Optimal-Trace Distillation.

Specialist Host-Verified Banks (all backed by ATMS SQLite ledger with drift invalidation):
  1. BANK_SCHEMA_JOINS: Active ATMS schema + verified JOIN edge graph + verbatim column value-format samples.
  2. BANK_OPTIMAL_TRACES: Minimal host-verified 1-to-2-step optimal traces from CORRECT! instances
     (answer-producing SQL + prerequisite lookup step if referenced + verbatim rows + answer format).
  3. BANK_ERROR_REPAIRS: Host-observed (failing SQL -> SQLite error -> corrected SQL) pairs + capped wrong answers.
  4. BANK_DRIFT_MIGRATION: Post-drift table/column diff + adapted pre-drift optimal-trace templates.

Deterministic Gating Router (`route_banks`):
  Computes question-conditioned weights across banks (table/column overlap, SQL op cues, post-drift flag)
  and allocates the character budget without mid-line truncation (`has_truncated=0`).
  Modes:
    - `moe_turns`: renders routed banks as ICL-native alternating (user, assistant, user) turns + system schema.
    - `moe_system`: renders routed banks as a structured [MIXTURE_OF_ENGRAMS] system digest.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, ClassVar, Optional

from ...interface import Observation, Query, Response
from ...registry import register_system
from ..icl.system import ICLSystem
from ..ours_engram.system import (
    _DRIFT_NOTICE_MARKER,
    _HOST_CAPABILITY_TOKEN,
    _QUERY_RESULT_PREFIX_RE,
    _referenced_tables,
)
from ..ours_engram_v2.system import _compact_sql, _extract_compact_question
from ..ours_engram_v3.system import (
    HostSQLV3Engram,
    _SEP_LINE_RE,
    _TOKEN_RE,
    _TOTAL_ROWS_RE,
    _answer_in_output,
    _format_verbatim_rows,
    _overlap_score,
)

MODES = ("moe_turns", "moe_system", "moe_lazy_turns", "moe_lazy_system", "moe_gt_turns")

_GT_ANSWER_RE = re.compile(r"(?:Correct answer|The correct answer was):\s*(.+)", re.IGNORECASE)


def _extract_ground_truth(obs_text: str) -> str:
    """Extract the host-provided ground-truth answer from an instance_complete INCORRECT/TIMEOUT observation."""
    m = _GT_ANSWER_RE.search(obs_text or "")
    if not m:
        return ""
    return m.group(1).strip().splitlines()[0].strip()[:160]

_JOIN_ON_RE = re.compile(
    r"\bJOIN\s+([`\"\[\]]?[A-Za-z_][A-Za-z0-9_]*[`\"\[\]]?)(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?\s+ON\s+([^\n;]{3,100}?)(?=\s+(?:WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|GROUP|ORDER|LIMIT|HAVING)\b|$)",
    re.IGNORECASE,
)
_SQL_OP_CUES = {
    "count": ("how many", "count", "number of", "total number"),
    "sum": ("total", "sum", "revenue", "amount", "spend", "sales"),
    "avg": ("average", "avg", "mean"),
    "extrema": ("highest", "lowest", "most", "least", "maximum", "minimum", "top", "largest", "smallest"),
    "group": ("each", "per ", "by ", "breakdown", "group"),
    "distinct": ("unique", "distinct", "different"),
    "date": ("year", "month", "date", "between", "after", "before", "quarter", "202", "201"),
}


def _question_op_cues(question: str) -> set[str]:
    low = (question or "").lower()
    return {op for op, kws in _SQL_OP_CUES.items() if any(k in low for k in kws)}


def _sql_op_cues(sql: str) -> set[str]:
    up = (sql or "").upper()
    out: set[str] = set()
    if "COUNT(" in up:
        out.add("count")
    if "SUM(" in up:
        out.add("sum")
    if "AVG(" in up:
        out.add("avg")
    if "MAX(" in up or "MIN(" in up or "ORDER BY" in up or "LIMIT" in up:
        out.add("extrema")
    if "GROUP BY" in up:
        out.add("group")
    if "DISTINCT" in up:
        out.add("distinct")
    if "DATE" in up or "STRFTIME" in up or "BETWEEN" in up or "LIKE '20" in up:
        out.add("date")
    return out


def _extract_value_samples(raw_obs: str, dep_tables: list[str]) -> list[tuple[str, str, str]]:
    """Extract verbatim (table_or_any, col_name, sample_val) pairs from a pipe-formatted SQLite result."""
    body = re.sub(r"^Query result\s*(?:\([^)]*\))?:\s*\n?", "", (raw_obs or "").strip(), flags=re.IGNORECASE).strip()
    body = _QUERY_RESULT_PREFIX_RE.sub("", body).strip()
    if not body or body.startswith("ERROR:"):
        return []
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip() and not _TOTAL_ROWS_RE.search(ln)]
    if len(lines) < 2 or "|" not in lines[0]:
        return []
    cols = [c.strip() for c in lines[0].strip("|").split("|")]
    data_start = 2 if (len(lines) > 2 and _SEP_LINE_RE.match(lines[1])) else 1
    if data_start >= len(lines):
        return []
    tbl = dep_tables[0] if len(dep_tables) == 1 else "*"
    samples: list[tuple[str, str, str]] = []
    seen_cols: set[str] = set()
    for row_ln in lines[data_start : data_start + 2]:
        if "|" not in row_ln:
            continue
        vals = [v.strip() for v in row_ln.strip("|").split("|")]
        if len(vals) != len(cols):
            continue
        for c, v in zip(cols, vals):
            if not c or not v or v.upper() == "NULL" or len(v) > 36 or c in seen_cols:
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c):
                samples.append((tbl, c, v))
                seen_cols.add(c)
    return samples[:6]


def _answer_in_output_relaxed(answer: str, raw_obs: str) -> bool:
    """Host-verified match allowing standard rounding (2dp), pct (0.153 <-> 15.3%), and multi-token rows."""
    if _answer_in_output(answer, raw_obs):
        return True
    ans = (answer or "").strip()
    if not ans or not raw_obs:
        return False
    body = re.sub(r"^Query result\s*(?:\([^)]*\))?:\s*\n?", "", raw_obs.strip(), flags=re.IGNORECASE).strip()
    body = _QUERY_RESULT_PREFIX_RE.sub("", body).strip()
    if not body or body.startswith("ERROR:"):
        return False
    low_body = body.lower()
    cleaned = re.sub(r"[\$,%\s]", "", ans.lower())
    # Numeric rounding & percentage equivalence
    try:
        ans_f = float(cleaned)
        cands = [ans_f]
        if "%" in ans:
            cands.append(ans_f / 100.0)
        elif 0.0 < abs(ans_f) <= 100.0:
            cands.append(ans_f / 100.0)
        for tok in re.findall(r"-?\d+(?:\.\d+)?", body):
            try:
                tf = float(tok)
                for cf in cands:
                    if abs(tf - cf) <= max(0.015, 1e-3 * abs(cf)):
                        return True
                    if round(tf, 2) == round(cf, 2) or round(tf, 1) == round(cf, 1):
                        if abs(tf - cf) <= max(0.06, 5e-3 * abs(cf)):
                            return True
            except ValueError:
                pass
    except ValueError:
        pass
    # Multi-token match (all meaningful tokens >=3 chars or numbers appear in output)
    toks = [t for t in re.findall(r"[a-z0-9_]+", ans.lower()) if len(t) >= 3 or t.isdigit()]
    if len(toks) >= 2 and all(t in low_body for t in toks):
        return True
    return False


def _is_meta_query(sql: str) -> bool:
    low = (sql or "").strip().lower()
    return (
        low.startswith(".tables")
        or low.startswith(".schema")
        or low.startswith("pragma ")
        or "sqlite_master" in low
        or "sqlite_schema" in low
    )


def _extract_minimal_trace(
    final_answer: str,
    queries_log: list[tuple[str, str]],
    *,
    relaxed: bool = False,
) -> tuple[list[tuple[str, str]], bool]:
    """Distill a multi-query trajectory into a minimal 1-to-2-step host-verified optimal trace."""
    matcher = _answer_in_output_relaxed if relaxed else _answer_in_output
    ans_idx: Optional[int] = None
    for i in range(len(queries_log) - 1, -1, -1):
        if matcher(final_answer, queries_log[i][1]):
            ans_idx = i
            break
    if ans_idx is None and relaxed:
        # Fallback for CORRECT! instances where final answer was derived by arithmetic over returned rows
        for i in range(len(queries_log) - 1, -1, -1):
            if not _is_meta_query(queries_log[i][0]):
                ans_idx = i
                break
    if ans_idx is None:
        return [], False

    ans_sql, ans_obs = queries_log[ans_idx]
    ans_fmt, _ = _format_verbatim_rows(ans_obs, max_rows=3)
    steps: list[tuple[str, str]] = [(_compact_sql(ans_sql, limit=340), ans_fmt)]

    # Check if an earlier query produced a literal string/ID used inside ans_sql (2-step optimal trace)
    literals = set(re.findall(r"'([^']{2,40})'", ans_sql))
    if literals and ans_idx > 0:
        for j in range(ans_idx - 1, -1, -1):
            prev_sql, prev_obs = queries_log[j]
            if any(lit in prev_obs for lit in literals):
                prev_fmt, _ = _format_verbatim_rows(prev_obs, max_rows=2)
                if prev_fmt:
                    steps.insert(0, (_compact_sql(prev_sql, limit=260), prev_fmt))
                break
    return steps, True


class MixtureOfEngramsStore(HostSQLV3Engram):
    """4-Bank Mixture-of-Engrams ATMS ledger."""

    def __init__(self, *, lazy_drift: bool = False) -> None:
        super().__init__()
        self.lazy_drift = lazy_drift
        self._conn.executescript(
            """
            CREATE TABLE join_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                edge_text TEXT NOT NULL UNIQUE,
                dep_tables_json TEXT NOT NULL
            );
            CREATE TABLE value_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                table_name TEXT NOT NULL,
                col_name TEXT NOT NULL,
                sample_val TEXT NOT NULL,
                UNIQUE(table_name, col_name, sample_val)
            );
            CREATE TABLE optimal_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                question TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                final_answer TEXT NOT NULL,
                dep_tables_json TEXT NOT NULL,
                op_cues_json TEXT NOT NULL
            );
            CREATE TABLE gt_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                question TEXT NOT NULL,
                wrong_answer TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                dep_tables_json TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        self._drift_epoch: int = 0
        self._pre_drift_tables: set[str] = set()
        self._post_drift_tables: set[str] = set()
        self.stats.update(
            {
                "join_edges_recorded": 0,
                "value_samples_recorded": 0,
                "optimal_traces_recorded": 0,
                "optimal_traces_2step": 0,
                "gt_feedback_recorded": 0,
                "gt_feedback_matched_sql": 0,
                "drift_epochs": 0,
            }
        )

    def observe_sql_result(self, sql: str, obs_text: str) -> None:
        super().observe_sql_result(sql, obs_text)
        body = re.sub(r"^Query result\s*(?:\([^)]*\))?:\s*\n?", "", (obs_text or "").strip(), flags=re.IGNORECASE).strip()
        body = _QUERY_RESULT_PREFIX_RE.sub("", body).strip()
        if body.startswith("ERROR:") or "no such table:" in body.lower() or "no such column:" in body.lower():
            # Surgical lazy invalidation: retract traces/joins/samples ONLY for the broken table
            m_tbl = re.search(r"no such table:\s*([A-Za-z_][A-Za-z0-9_]*)", body, re.IGNORECASE)
            if m_tbl:
                self._stale_moe_touching(m_tbl.group(1))
            elif re.search(r"no such column:", body, re.IGNORECASE):
                for t in _referenced_tables(sql):
                    self._stale_moe_touching(t)
            return
        if not body:
            return
        deps = sorted(set(_referenced_tables(sql)))
        if self._drift_epoch == 0:
            self._pre_drift_tables.update(deps)
        else:
            self._post_drift_tables.update(deps)
            # Re-verify UNVERIFIED_POST_DRIFT items touching tables that just succeeded post-drift!
            if self.lazy_drift and deps:
                for t in deps:
                    self._reverify_moe_touching(t)

        # Harvest verified JOIN edges from non-error queries
        for m in _JOIN_ON_RE.finditer(sql):
            rhs_tbl = m.group(1).strip('`"[]')
            cond = " ".join(m.group(3).split())[:90]
            edge = f"JOIN {rhs_tbl} ON {cond}"
            edge_deps = sorted(set(deps) | {rhs_tbl.lower()})
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO join_edges (status, edge_text, dep_tables_json) VALUES ('ACTIVE', ?, ?)",
                (edge, json.dumps(edge_deps)),
            )
            if cur.rowcount:
                self.stats["join_edges_recorded"] += 1

        # Harvest verbatim column value-format samples
        for tbl, col, val in _extract_value_samples(obs_text, deps):
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO value_samples (status, table_name, col_name, sample_val) VALUES ('ACTIVE', ?, ?, ?)",
                (tbl, col, val),
            )
            if cur.rowcount:
                self.stats["value_samples_recorded"] += 1
        self._conn.commit()

    def _stale_moe_touching(self, table: str) -> None:
        t_low = table.lower()
        for tbl in ("join_edges", "optimal_traces", "gt_feedback"):
            for r in self._conn.execute(
                f"SELECT id, dep_tables_json FROM {tbl} WHERE status IN ('ACTIVE', 'UNVERIFIED_POST_DRIFT')"
            ).fetchall():
                deps = [str(x).lower() for x in json.loads(r["dep_tables_json"])]
                if t_low in deps:
                    self._conn.execute(f"UPDATE {tbl} SET status = 'SCHEMA_STALE' WHERE id = ?", (r["id"],))
        self._conn.execute(
            "UPDATE value_samples SET status = 'SCHEMA_STALE' WHERE LOWER(table_name) = ? AND status IN ('ACTIVE', 'UNVERIFIED_POST_DRIFT')",
            (t_low,),
        )
        self._conn.commit()

    def _reverify_moe_touching(self, table: str) -> None:
        t_low = table.lower()
        for tbl in ("join_edges", "optimal_traces", "gt_feedback"):
            for r in self._conn.execute(
                f"SELECT id, dep_tables_json FROM {tbl} WHERE status = 'UNVERIFIED_POST_DRIFT'"
            ).fetchall():
                deps = [str(x).lower() for x in json.loads(r["dep_tables_json"])]
                if deps and all(d in {x.lower() for x in self._post_drift_tables} for d in deps):
                    self._conn.execute(f"UPDATE {tbl} SET status = 'ACTIVE' WHERE id = ?", (r["id"],))
        self._conn.execute(
            "UPDATE value_samples SET status = 'ACTIVE' WHERE LOWER(table_name) = ? AND status = 'UNVERIFIED_POST_DRIFT'",
            (t_low,),
        )
        self._conn.commit()

    def stale_all_for_drift(self, *, reason: str, capability: object) -> None:
        if capability is not _HOST_CAPABILITY_TOKEN:
            raise PermissionError("Host capability token required")
        self._drift_epoch += 1
        self.stats["drift_epochs"] = self._drift_epoch
        if self.lazy_drift:
            # Lazy ATMS drift: keep pre-drift schema & optimal traces visible as UNVERIFIED_POST_DRIFT
            # until surgical per-table error/supersede retracts only the actually changed tables!
            self.stats["drift_resets"] += 1
            for tbl in ("join_edges", "value_samples", "optimal_traces"):
                self._conn.execute(f"UPDATE {tbl} SET status = 'UNVERIFIED_POST_DRIFT' WHERE status = 'ACTIVE'")
            # Only table-dependent gt_feedback transitions to UNVERIFIED_POST_DRIFT; format-only notes stay ACTIVE
            self._conn.execute(
                "UPDATE gt_feedback SET status = 'UNVERIFIED_POST_DRIFT' WHERE status = 'ACTIVE' AND dep_tables_json != '[]'"
            )
            self._conn.commit()
            return
        super().stale_all_for_drift(reason=reason, capability=capability)
        for tbl in ("join_edges", "value_samples", "optimal_traces", "gt_feedback"):
            self._conn.execute(f"UPDATE {tbl} SET status = 'SCHEMA_STALE' WHERE status = 'ACTIVE'")
        self._conn.commit()

    def commit_optimal_trace(
        self,
        *,
        correct: bool,
        question: str,
        queries_log: list[tuple[str, str]],
        final_answer: str,
        capability: object,
        obs_text: str = "",
    ) -> None:
        if capability is not _HOST_CAPABILITY_TOKEN:
            raise PermissionError("Host capability token required")
        # Keep base v3 failed/confirmed card tracking in sync
        self.commit_verified_case(
            correct=correct,
            question=question,
            queries_log=queries_log,
            final_answer=final_answer,
            capability=capability,
        )
        if not question:
            return
        if not correct:
            gt = _extract_ground_truth(obs_text)
            if gt:
                # Check strictly (no arbitrary fallback) if any query output already matched gt!
                steps: list[tuple[str, str]] = []
                deps: list[str] = []
                for i in range(len(queries_log) - 1, -1, -1):
                    if _answer_in_output_relaxed(gt, queries_log[i][1]):
                        ans_sql, ans_obs = queries_log[i]
                        ans_fmt, _ = _format_verbatim_rows(ans_obs, max_rows=3)
                        steps = [(_compact_sql(ans_sql, limit=340), ans_fmt)]
                        deps = sorted(set(_referenced_tables(ans_sql)))
                        self.stats["gt_feedback_matched_sql"] += 1
                        break
                self._conn.execute(
                    """
                    INSERT INTO gt_feedback (status, question, wrong_answer, correct_answer, steps_json, dep_tables_json)
                    VALUES ('ACTIVE', ?, ?, ?, ?, ?)
                    """,
                    (question, (final_answer or "(none)")[:120], gt, json.dumps(steps), json.dumps(deps)),
                )
                self.stats["gt_feedback_recorded"] += 1
                self._conn.commit()
            return
        steps, ok = _extract_minimal_trace(final_answer, queries_log, relaxed=self.lazy_drift)
        if not ok:
            return
        all_sql = " ".join(s for s, _ in steps)
        deps = sorted(set(_referenced_tables(all_sql)))
        cues = sorted(_sql_op_cues(all_sql))
        self._conn.execute(
            """
            INSERT INTO optimal_traces (status, question, steps_json, final_answer, dep_tables_json, op_cues_json)
            VALUES ('ACTIVE', ?, ?, ?, ?, ?)
            """,
            (question, json.dumps(steps), (final_answer or "")[:120], json.dumps(deps), json.dumps(cues)),
        )
        self.stats["optimal_traces_recorded"] += 1
        if len(steps) > 1:
            self.stats["optimal_traces_2step"] += 1
        self._conn.commit()

    def route_banks(
        self,
        *,
        current_question: str,
        max_chars: int = 12000,
    ) -> dict[str, Any]:
        """Deterministically route and budget all 4 specialist Engram banks for `current_question`."""
        q_cues = _question_op_cues(current_question)
        q_toks = set(_TOKEN_RE.findall((current_question or "").lower()))

        # Bank 1: Schema + verified JOIN edges + verbatim value-format samples
        schema_blk = self.render_engram_block(max_chars=min(4200, max_chars // 3))
        if self.lazy_drift and self._drift_epoch > 0 and schema_blk:
            schema_blk = (
                "[DRIFT_NOTICE: Schema migration occurred; entries below are preserved until contradicted "
                "by a post-drift error or new PRAGMA/SELECT — verify if a table/column changed.]\n" + schema_blk
            )
        active_st = "('ACTIVE', 'UNVERIFIED_POST_DRIFT')" if self.lazy_drift else "('ACTIVE')"
        active_joins = self._conn.execute(f"SELECT * FROM join_edges WHERE status IN {active_st} ORDER BY id ASC").fetchall()
        active_vals = self._conn.execute(f"SELECT * FROM value_samples WHERE status IN {active_st} ORDER BY id DESC LIMIT 24").fetchall()
        active_traces = self._conn.execute(f"SELECT * FROM optimal_traces WHERE status IN {active_st} ORDER BY id ASC").fetchall()
        stale_traces = self._conn.execute("SELECT * FROM optimal_traces WHERE status = 'SCHEMA_STALE' ORDER BY id DESC").fetchall()
        active_errors = self._conn.execute("SELECT * FROM error_pairs WHERE status = 'ACTIVE' ORDER BY id DESC").fetchall()
        failed_active = self._conn.execute(
            "SELECT * FROM case_cards WHERE lane = 'OUTCOME_FAILED' AND status = 'ACTIVE' ORDER BY id DESC"
        ).fetchall()

        # Score optimal traces by (table/SQL overlap + SQL op-cue match, recency)
        def _trace_rank(r: Any) -> tuple[int, int]:
            steps = json.loads(r["steps_json"])
            all_sql = " ".join(s for s, _ in steps)
            base = _overlap_score(current_question, r["question"], json.loads(r["dep_tables_json"]), all_sql)
            cue_bonus = 2 * len(q_cues & set(json.loads(r["op_cues_json"])))
            return (base + cue_bonus, int(r["id"]))

        ranked_traces = sorted(active_traces, key=_trace_rank, reverse=True)
        ranked_stale = sorted(stale_traces, key=_trace_rank, reverse=True)
        ranked_errs = sorted(
            active_errors,
            key=lambda r: (_overlap_score(current_question, "", json.loads(r["dep_tables_json"]), r["fixed_sql"]), r["id"]),
            reverse=True,
        )

        # Compute deterministic expert weights for telemetry + dynamic sub-budgets
        w_traces = 0.55 if ranked_traces else 0.0
        w_migration = 0.25 if (self._drift_epoch > 0 and ranked_stale) else 0.0
        w_repairs = 0.20 if ranked_errs else 0.0
        w_schema = 1.0 - min(0.85, w_traces + w_migration + w_repairs)

        used = len(schema_blk)
        rem = max(0, max_chars - used - 500)

        # Bank 1 enrichments: JOIN graph + value samples (compact, <= 1200 chars)
        join_lines: list[str] = []
        for j in active_joins[:10]:
            ln = f"  - {j['edge_text']}"
            if len(ln) + 1 <= min(600, rem):
                join_lines.append(ln)
                rem -= len(ln) + 1

        val_lines: list[str] = []
        # Prefer columns matching question tokens first
        sorted_vals = sorted(
            active_vals,
            key=lambda r: (1 if (r["col_name"].lower() in q_toks or r["table_name"].lower() in q_toks) else 0, r["id"]),
            reverse=True,
        )
        seen_vc: set[tuple[str, str]] = set()
        for v in sorted_vals:
            key = (v["table_name"], v["col_name"])
            if key in seen_vc or len(val_lines) >= 12:
                continue
            seen_vc.add(key)
            ln = f"{v['table_name']}.{v['col_name']}='{v['sample_val']}'"
            if len(ln) + 2 <= min(600, rem):
                val_lines.append(ln)
                rem -= len(ln) + 2

        # Bank 2: Optimal Traces (highest priority for remaining budget)
        picked_traces: list[dict[str, Any]] = []
        trace_budget = int(rem * (0.72 if not w_migration else 0.55))
        if not ranked_errs and not w_migration:
            trace_budget = rem
        for r in ranked_traces:
            steps = json.loads(r["steps_json"])
            cost = len(r["question"]) + len(r["final_answer"]) + sum(len(s) + len(o) + 40 for s, o in steps) + 50
            if cost <= trace_budget:
                d = dict(r)
                d["steps"] = steps
                picked_traces.append(d)
                trace_budget -= cost
                rem -= cost

        # Bank 4: Post-Drift Migration Bank (only when drifted)
        picked_migration: list[dict[str, Any]] = []
        if self._drift_epoch > 0 and ranked_stale:
            mig_budget = min(rem, 2200)
            for r in ranked_stale[:6]:
                steps = json.loads(r["steps_json"])
                last_sql = steps[-1][0] if steps else ""
                cost = len(r["question"]) + len(last_sql) + len(r["final_answer"]) + 60
                if cost <= mig_budget:
                    d = dict(r)
                    d["last_sql"] = last_sql
                    picked_migration.append(d)
                    mig_budget -= cost
                    rem -= cost

        # Bank 3: Error Repairs + capped failed warnings
        picked_errors: list[dict[str, Any]] = []
        for r in ranked_errs[:8]:
            cost = len(r["bad_sql"]) + len(r["error_msg"]) + len(r["fixed_sql"]) + 50
            if cost <= rem:
                picked_errors.append(dict(r))
                rem -= cost

        picked_failed: list[dict[str, Any]] = []
        for r in failed_active[: min(len(picked_traces), 3)]:
            cost = len(r["question"]) + len(r["final_answer"]) + 40
            if cost <= rem:
                picked_failed.append(dict(r))
                rem -= cost

        # Chronological ordering inside each bank maximizes KV prefix reuse across instances!
        picked_traces.sort(key=lambda r: r["id"])
        picked_migration.sort(key=lambda r: r["id"])
        picked_errors.sort(key=lambda r: r["id"])
        picked_failed.sort(key=lambda r: r["id"])

        return {
            "schema_blk": schema_blk,
            "join_lines": join_lines,
            "val_lines": val_lines,
            "traces": picked_traces,
            "migration": picked_migration,
            "errors": picked_errors,
            "failed": picked_failed,
            "weights": {
                "schema": round(w_schema, 2),
                "traces": round(w_traces, 2),
                "repairs": round(w_repairs, 2),
                "migration": round(w_migration, 2),
            },
            "drift_delta": {
                "pre_only": sorted(self._pre_drift_tables - self._post_drift_tables) if self._drift_epoch > 0 else [],
                "post_new": sorted(self._post_drift_tables - self._pre_drift_tables) if self._drift_epoch > 0 else [],
            },
        }

    def render_moe(
        self,
        *,
        current_question: str,
        mode: str,
        max_chars: int = 12000,
    ) -> tuple[str, list[dict[str, str]], dict[str, float]]:
        sel = self.route_banks(current_question=current_question, max_chars=max_chars)

        sys_parts: list[str] = []
        if sel["schema_blk"]:
            sys_parts.append(sel["schema_blk"])
        if sel["join_lines"] or sel["val_lines"]:
            b1 = ["[ENGRAM_BANK_1: VERIFIED_JOINS_AND_VALUE_FORMATS]"]
            if sel["join_lines"]:
                b1.append("VERIFIED_JOIN_EDGES:")
                b1.extend(sel["join_lines"])
            if sel["val_lines"]:
                b1.append("OBSERVED_VALUE_FORMATS: " + "; ".join(sel["val_lines"]))
            b1.append("[/ENGRAM_BANK_1]")
            sys_parts.append("\n".join(b1))

        if sel["migration"]:
            b4 = ["[ENGRAM_BANK_4: POST_DRIFT_MIGRATION_ADAPTER]"]
            dd = sel["drift_delta"]
            if dd["pre_only"] or dd["post_new"]:
                b4.append(f"OBSERVED_TABLE_DELTA: removed/renamed={dd['pre_only']} | new_active={dd['post_new']}")
            b4.append("PRIOR_OPTIMAL_SQL_TEMPLATES (re-map to new active schema before running):")
            for m in sel["migration"]:
                b4.append(f"  - Q: {m['question']} | PriorSQL: {m['last_sql']} | AnsFmt: {m['final_answer']}")
            b4.append("[/ENGRAM_BANK_4]")
            sys_parts.append("\n".join(b4))

        if mode == "moe_gt_turns":
            # Positive optimal traces + truthful host GT feedback (ZERO error_pairs, ZERO zero-query turns!)
            active_st = "('ACTIVE', 'UNVERIFIED_POST_DRIFT')" if self.lazy_drift else "('ACTIVE')"
            gt_rows = self._conn.execute(
                f"SELECT * FROM gt_feedback WHERE status IN {active_st} ORDER BY id DESC LIMIT 16"
            ).fetchall()
            gt_with_sql = [dict(r) for r in gt_rows if json.loads(r["steps_json"])]
            gt_fmt_only = [dict(r) for r in gt_rows if not json.loads(r["steps_json"])]
            gt_with_sql.sort(key=lambda r: r["id"])
            gt_fmt_only = sorted(gt_fmt_only[:8], key=lambda r: r["id"])

            if gt_fmt_only:
                b3 = [
                    "[ENGRAM_BANK_3: HOST_ANSWER_FORMAT_FEEDBACK]",
                    "Prior scored questions (learn exact formatting, rounding, units, and conventions; NO SQL omitted):",
                ]
                for g in gt_fmt_only:
                    b3.append(
                        f"  - Q: {g['question']} | YourAns: {g['wrong_answer']} -> CorrectAns: {g['correct_answer']}"
                    )
                b3.append("[/ENGRAM_BANK_3]")
                sys_parts.append("\n".join(b3))

            turns = []
            # Interleave optimal traces & matched-SQL GT corrections in chronological order for KV prefix reuse
            timeline: list[tuple[int, str, dict[str, Any]]] = []
            for t in sel["traces"]:
                timeline.append((int(t["id"]), "trace", t))
            for g in gt_with_sql[:6]:
                g["steps"] = json.loads(g["steps_json"])
                timeline.append((100000 + int(g["id"]), "gt_sql", g))
            for _, kind, item in timeline:
                if kind == "trace":
                    turns.append({"role": "user", "content": f"Question: {item['question']}"})
                    for sql_s, res_s in item["steps"]:
                        turns.append({"role": "assistant", "content": json.dumps({"action": "QUERY", "content": sql_s})})
                        turns.append({"role": "user", "content": f"FEEDBACK: Query result:\n{res_s}"})
                    turns.append({"role": "assistant", "content": json.dumps({"action": "ANSWER", "content": item["final_answer"]})})
                    turns.append({"role": "user", "content": "FEEDBACK: Outcome: CORRECT!"})
                else:
                    turns.append({"role": "user", "content": f"Question: {item['question']}"})
                    for sql_s, res_s in item["steps"]:
                        turns.append({"role": "assistant", "content": json.dumps({"action": "QUERY", "content": sql_s})})
                        turns.append({"role": "user", "content": f"FEEDBACK: Query result:\n{res_s}"})
                    turns.append({"role": "assistant", "content": json.dumps({"action": "ANSWER", "content": item["wrong_answer"]})})
                    turns.append(
                        {
                            "role": "user",
                            "content": (
                                f"FEEDBACK: Outcome: INCORRECT.\n"
                                f"Your answer: {item['wrong_answer']}\n"
                                f"Correct answer: {item['correct_answer']}"
                            ),
                        }
                    )
            return "\n\n".join(sys_parts), turns, sel["weights"]

        if mode in ("moe_turns", "moe_lazy_turns"):
            turns: list[dict[str, str]] = []
            for t in sel["traces"]:
                turns.append({"role": "user", "content": f"Question: {t['question']}"})
                for sql_s, res_s in t["steps"]:
                    turns.append({"role": "assistant", "content": json.dumps({"action": "QUERY", "content": sql_s})})
                    turns.append({"role": "user", "content": f"Query result:\n{res_s}"})
                turns.append({"role": "assistant", "content": json.dumps({"action": "ANSWER", "content": t["final_answer"]})})
                turns.append({"role": "user", "content": "Outcome: CORRECT!"})
            for e in sel["errors"]:
                if not turns or turns[-1]["role"] != "user":
                    turns.append({"role": "user", "content": "Prior SQL error repair:"})
                turns.append({"role": "assistant", "content": json.dumps({"action": "QUERY", "content": e["bad_sql"]})})
                turns.append({"role": "user", "content": f"Query result:\n{e['error_msg']}"})
                turns.append({"role": "assistant", "content": json.dumps({"action": "QUERY", "content": e["fixed_sql"]})})
                turns.append({"role": "user", "content": "Query result: succeeded."})
            for f in sel["failed"]:
                turns.append({"role": "user", "content": f"Question: {f['question']}"})
                turns.append({"role": "assistant", "content": json.dumps({"action": "ANSWER", "content": f["final_answer"]})})
                turns.append({"role": "user", "content": "Outcome: INCORRECT"})
            return "\n\n".join(sys_parts), turns, sel["weights"]

        # mode == "moe_system"
        if sel["traces"] or sel["errors"] or sel["failed"]:
            b23 = ["[MIXTURE_OF_ENGRAMS_BANKS_2_3]"]
            if sel["traces"]:
                b23.append("BANK_2_OPTIMAL_TRACES (host-verified minimal SQL + verbatim rows):")
                for t in sel["traces"]:
                    b23.append(f"  - Q: {t['question']}")
                    for idx_s, (sql_s, res_s) in enumerate(t["steps"], 1):
                        b23.append(f"    STEP{idx_s}_SQL: {sql_s}")
                        if res_s:
                            ind = "\n".join("      " + ln for ln in res_s.splitlines())
                            b23.append(f"    STEP{idx_s}_ROWS:\n{ind}")
                    b23.append(f"    VERIFIED_ANSWER: {t['final_answer']}")
            if sel["errors"]:
                b23.append("BANK_3_ERROR_REPAIRS (observed fail -> working fix):")
                for e in sel["errors"]:
                    b23.append(f"  - FAIL: {e['bad_sql']} -> {e['error_msg']} | FIX: {e['fixed_sql']}")
            if sel["failed"]:
                b23.append("BANK_3_WRONG_ANSWERS (avoid):")
                for f in sel["failed"]:
                    b23.append(f"  - Q: {f['question']} | WrongAns: {f['final_answer']} (INCORRECT)")
            b23.append("[/MIXTURE_OF_ENGRAMS_BANKS_2_3]")
            sys_parts.append("\n".join(b23))
        return "\n\n".join(sys_parts), [], sel["weights"]


@register_system("ours_engram_moe")
class OursEngramMoESystem(ICLSystem):
    """Fresh-session CL-Bench system backed by deterministic 4-Bank Mixture of Engrams."""

    supports_baseline: bool = False
    parallel_safe: ClassVar[bool] = True

    def __init__(
        self,
        model: str = "gpt-5",
        max_tokens: int | None = None,
        system_prompt: str = "",
        name: str = "ours_engram_moe",
        reserve_tokens: int = 500,
        provider_mode: str = "litellm_chat",
        mode: str = "moe_turns",
        max_engram_chars: int = 12000,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            name=name,
            reserve_tokens=reserve_tokens,
            provider_mode=provider_mode,
        )
        self.mode = mode
        self.lazy_drift = mode in ("moe_lazy_turns", "moe_lazy_system", "moe_gt_turns")
        if max_engram_chars == 12000:
            if mode == "moe_gt_turns":
                self.max_engram_chars = 18000
            elif self.lazy_drift:
                self.max_engram_chars = 16000
            else:
                self.max_engram_chars = max_engram_chars
        else:
            self.max_engram_chars = max_engram_chars
        self._store = MixtureOfEngramsStore(lazy_drift=self.lazy_drift)
        self._rendered_sys: str = ""
        self._rendered_turns: list[dict[str, str]] = []
        self._last_weights: dict[str, float] = {}
        self._rendered_for_instance: Optional[str] = None
        self._drift_handled_for_instance: Optional[str] = None
        self._inst_question: str = ""
        self._last_sql: Optional[str] = None
        self._pending_error: Optional[tuple[str, str]] = None
        self._inst_queries: list[tuple[str, str]] = []
        self._final_answer: str = ""
        self._instance_truncated: bool = False
        self._stats_path = os.environ.get("OURS_ENGRAM_STATS_LOG")

    def reset(self) -> None:
        super().reset()
        self._store.close()
        self._store = MixtureOfEngramsStore(lazy_drift=self.lazy_drift)
        self._rendered_sys = ""
        self._rendered_turns = []
        self._last_weights = {}
        self._rendered_for_instance = None
        self._drift_handled_for_instance = None
        self._inst_question = ""
        self._last_sql = None
        self._pending_error = None
        self._inst_queries = []
        self._final_answer = ""
        self._instance_truncated = False

    def _system_messages(self) -> list[dict]:
        base = list(super()._system_messages())
        if self._rendered_sys:
            base.append({"role": "system", "content": self._rendered_sys})
        if self._rendered_turns:
            base.extend(self._rendered_turns)
        return base

    def _note_prompt_token_usage(self, input_tokens: int | None) -> None:
        self._token_budget.note_usage(
            messages=[*self._system_messages(), *self.messages],
            input_tokens=input_tokens,
            estimate_fn=self._estimate_message_tokens,
        )

    def _truncate_context(self, *args: Any, **kwargs: Any) -> None:
        before = len(self.messages)
        super()._truncate_context(*args, **kwargs)
        if len(self.messages) < before:
            self._instance_truncated = True

    def respond(self, query: Query) -> Response:
        inst_key = str(query.instance_id or f"idx:{query.instance_index}")
        is_new_instance = self._rendered_for_instance != inst_key

        if is_new_instance:
            self._inst_question = _extract_compact_question(query.prompt or "")
            self._last_sql = None
            self._pending_error = None
            self._inst_queries = []
            self._final_answer = ""
            self._instance_truncated = False

        drift_fired = False
        if (
            query.prompt
            and _DRIFT_NOTICE_MARKER in query.prompt.lower()
            and self._drift_handled_for_instance != inst_key
        ):
            self._store.stale_all_for_drift(
                reason="host_schema_drift_notice", capability=_HOST_CAPABILITY_TOKEN
            )
            self._drift_handled_for_instance = inst_key
            drift_fired = True

        if is_new_instance or drift_fired:
            self._rendered_sys, self._rendered_turns, self._last_weights = self._store.render_moe(
                current_question=self._inst_question,
                mode=self.mode,
                max_chars=self.max_engram_chars,
            )
            self._rendered_for_instance = inst_key

        response = super().respond(query)
        action_obj = response.action
        self._last_sql = None
        if action_obj is not None:
            act = getattr(action_obj, "action", None)
            content = getattr(action_obj, "content", None)
            if isinstance(act, str) and isinstance(content, str):
                act_u = act.strip().upper()
                if act_u == "QUERY":
                    self._last_sql = content
                elif act_u == "ANSWER":
                    self._final_answer = content.strip()
        return response

    def observe(self, observation: Observation, next_query: Optional[Query] = None) -> None:
        text = observation.content or ""
        if not observation.instance_complete:
            if self._last_sql is not None:
                self._store.observe_sql_result(self._last_sql, text)
                body = _QUERY_RESULT_PREFIX_RE.sub("", text.strip()).strip()
                if body.startswith("ERROR:"):
                    self._pending_error = (self._last_sql, body.splitlines()[0])
                elif body and body != "(no results)":
                    self._inst_queries.append((self._last_sql, text))
                    if self._pending_error is not None:
                        bad_sql, err_line = self._pending_error
                        self._store.commit_error_pair(
                            bad_sql=bad_sql,
                            error_msg=err_line,
                            fixed_sql=self._last_sql,
                            capability=_HOST_CAPABILITY_TOKEN,
                        )
                        self._pending_error = None
                self._last_sql = None
            super().observe(observation, next_query)
            return

        is_correct = ": CORRECT!" in text
        self._store.commit_optimal_trace(
            correct=is_correct,
            question=self._inst_question,
            queries_log=self._inst_queries,
            final_answer=self._final_answer,
            capability=_HOST_CAPABILITY_TOKEN,
            obs_text=text,
        )

        self._last_sql = None
        self._pending_error = None
        self._inst_queries = []
        self._rendered_for_instance = None

        if self._stats_path:
            try:
                os.makedirs(os.path.dirname(self._stats_path), exist_ok=True)
                with open(self._stats_path, "a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "t": round(time.time(), 3),
                                "pid": os.getpid(),
                                "mode": self.mode,
                                "engram_chars": len(self._rendered_sys)
                                + sum(len(m["content"]) for m in self._rendered_turns),
                                "has_truncated": self._instance_truncated,
                                "weights": self._last_weights,
                                **self._store.stats,
                            }
                        )
                        + "\n"
                    )
            except OSError:
                pass

        self._instance_truncated = False
        ICLSystem.reset(self)
