"""KEGG pathway over-representation analysis using the official KEGG REST API.

KEGG REST does not provide a one-shot enrichment endpoint. This module downloads
and caches human gene/pathway annotations, then runs a hypergeometric ORA locally
and applies Benjamini-Hochberg FDR correction.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom


def _norm(value: Any) -> str:
    return str(value).strip().upper()


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


class KEGGEnrichmentRunner:
    """Drop-in enrichment runner compatible with the ToppFun transfer workflow."""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = dict(config or {})
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / "cache"
        self.raw_dir = self.output_dir / "raw_json"
        self.request_dir = self.output_dir / "requests"
        self.input_dir = self.output_dir / "input_gene_lists"
        for p in (self.cache_dir, self.raw_dir, self.request_dir, self.input_dir):
            p.mkdir(parents=True, exist_ok=True)

        self.endpoint = str(self.config.get("endpoint", "https://rest.kegg.jp")).rstrip("/")
        self.organism = str(self.config.get("organism", "hsa")).lower()
        if self.organism != "hsa":
            raise ValueError("This implementation currently supports human TCGA cohorts only (organism='hsa').")
        self.timeout = float(self.config.get("timeout_seconds", 30))
        self.retries = max(1, int(self.config.get("retries", 2)))
        self.interval = max(1.0 / 3.0, float(self.config.get("request_interval_seconds", 0.4)))
        self.use_cache = bool(self.config.get("use_cache", True))
        self.background_mode = str(self.config.get("background", "all_category_genes"))
        self.top_genes = int(self.config.get("top_genes_per_module", 50))
        self.min_genes = max(1, int(self.config.get("min_genes", 2)))
        self.max_results = max(1, int(self.config.get("max_results", 100)))
        self.significance_qvalue = float(self.config.get("significance_qvalue", 0.05))
        self.variants = set(str(x) for x in self.config.get("variants", ["none"]))
        self.session = requests.Session()
        self.result_rows: List[Dict[str, Any]] = []
        self.status_rows: List[Dict[str, Any]] = []
        self.mapping_rows: List[Dict[str, Any]] = []
        self._symbol_to_kegg: Dict[str, str] = {}
        self._kegg_to_symbol: Dict[str, str] = {}
        self._pathway_to_genes: Dict[str, set] = {}
        self._pathway_names: Dict[str, str] = {}
        self._universe: set = set()
        self._load_or_fetch_annotations()

    def _get_text(self, operation: str, cache_name: str) -> str:
        cache_file = self.cache_dir / cache_name
        if self.use_cache and cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        url = f"{self.endpoint}/{operation.lstrip('/')}"
        request_record = {"method": "GET", "url": url, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (self.request_dir / f"{cache_name}.request.json").write_text(
            json.dumps(request_record, indent=2), encoding="utf-8"
        )
        last_error = None
        for attempt in range(self.retries):
            try:
                time.sleep(self.interval)
                started = time.time()
                print(f"[KEGG API] GET {url} attempt={attempt + 1}/{self.retries}", flush=True)
                response = self.session.get(url, timeout=(5, self.timeout))
                response.raise_for_status()
                text = response.text
                if not text.strip():
                    raise RuntimeError("KEGG returned an empty response")
                print(f"[KEGG API] HTTP {response.status_code} elapsed={time.time()-started:.2f}s", flush=True)
                (self.raw_dir / f"{cache_name}.txt").write_text(text, encoding="utf-8")
                if self.use_cache:
                    cache_file.write_text(text, encoding="utf-8")
                return text
            except Exception as exc:
                last_error = exc
                print(f"[KEGG API] failed: {type(exc).__name__}: {exc}", flush=True)
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"KEGG request failed ({url}): {last_error}")

    def _load_or_fetch_annotations(self) -> None:
        # These three relatively large tables are downloaded once and reused from cache.
        genes_text = self._get_text(f"list/{self.organism}", "list_hsa.txt")
        links_text = self._get_text(f"link/pathway/{self.organism}", "link_pathway_hsa.txt")
        pathway_text = self._get_text(f"list/pathway/{self.organism}", "list_pathway_hsa.txt")

        for line in genes_text.splitlines():
            fields = line.split("\t", 1)
            if len(fields) != 2 or ":" not in fields[0]:
                continue
            kid, description = fields[0].strip(), fields[1].strip()
            # KEGG descriptions commonly start with SYMBOL, aliases; description.
            name_part = description.split(" - ", 1)[0].split(";", 1)[0]
            # The first token is the canonical gene symbol; remaining text can be a full name.
            canonical = name_part.split(",", 1)[0].strip()
            if canonical:
                self._kegg_to_symbol[kid] = canonical
                self._symbol_to_kegg.setdefault(_norm(canonical), kid)

        for line in pathway_text.splitlines():
            fields = line.split("\t", 1)
            if len(fields) != 2:
                continue
            pid = fields[0].strip().replace("path:", "")
            if pid.startswith("hsa"):
                self._pathway_names[pid] = fields[1].strip()

        for line in links_text.splitlines():
            fields = line.split("\t", 1)
            if len(fields) != 2:
                continue
            gene_id, pathway_id = fields[0].strip(), fields[1].strip().replace("path:", "")
            if gene_id.startswith(f"{self.organism}:") and pathway_id in self._pathway_names:
                self._pathway_to_genes.setdefault(pathway_id, set()).add(gene_id)
        self._universe = set().union(*self._pathway_to_genes.values()) if self._pathway_to_genes else set()
        if not self._symbol_to_kegg or not self._pathway_to_genes:
            raise RuntimeError("KEGG annotation tables were retrieved but could not be parsed; check API format/cache files.")
        metadata = {
            "source": "KEGG REST API",
            "endpoint": self.endpoint,
            "organism": self.organism,
            "gene_symbols_mapped": len(self._symbol_to_kegg),
            "pathway_count": len(self._pathway_to_genes),
            "genes_in_pathway_universe": len(self._universe),
            "background_mode": self.background_mode,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (self.output_dir / "kegg_annotation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _map_symbols(self, symbols: Sequence[str], metadata: Dict[str, Any]) -> Tuple[set, List[Dict[str, Any]]]:
        ids = set()
        rows = []
        for symbol in dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()):
            kid = self._symbol_to_kegg.get(_norm(symbol))
            status = "mapped" if kid else "unmapped"
            if kid:
                ids.add(kid)
            rows.append({**metadata, "submitted_symbol": symbol, "kegg_gene_id": kid or "", "mapping_status": status})
        return ids, rows

    def enrich_one_module(self, gene_symbols: Sequence[Any], background_symbols: Sequence[Any], metadata: Dict[str, Any]):
        query_symbols = list(dict.fromkeys(str(x).strip() for x in gene_symbols if str(x).strip()))
        mapped_query_ids, mapping_rows = self._map_symbols(query_symbols, metadata)
        bg_symbols = list(dict.fromkeys(str(x).strip() for x in background_symbols if str(x).strip()))
        bg_ids, bg_mapping_rows = self._map_symbols(bg_symbols, metadata) if self.background_mode == "selected_gene_universe" else (set(), [])
        mapping_rows.extend(bg_mapping_rows)

        if self.background_mode == "selected_gene_universe":
            universe = bg_ids & self._universe
        else:
            universe = set(self._universe)
        query_ids = mapped_query_ids & universe
        n = len(query_ids)
        base_status = {
            **metadata,
            "input_gene_count": len(query_symbols),
            "mapped_query_gene_count": len(mapped_query_ids),
            "query_gene_count_in_background": n,
            "unmapped_query_gene_count": len(query_symbols) - len(mapped_query_ids),
            "background_symbol_count": len(bg_symbols) if self.background_mode == "selected_gene_universe" else 0,
            "mapped_background_gene_count": len(universe),
            "background_mode": self.background_mode,
            "term_count": 0,
            "status": "pending",
            "error": "",
        }
        if n < self.min_genes:
            return [], {**base_status, "status": "skipped_too_few_mapped_genes", "error": f"Only {n} query genes overlap KEGG background; min_genes={self.min_genes}"}, mapping_rows
        N = len(universe)
        if N == 0:
            return [], {**base_status, "status": "empty_background", "error": "No mapped genes in selected background"}, mapping_rows

        tested_rows = []
        for pathway_id, pathway_genes_all in self._pathway_to_genes.items():
            pathway_genes = pathway_genes_all & universe
            K = len(pathway_genes)
            if K == 0:
                continue
            overlap = query_ids & pathway_genes
            k = len(overlap)
            # Include zero-overlap pathways (p=1) in the BH family, then omit them from the report.
            p_value = float(hypergeom.sf(k - 1, N, K, n))
            tested_rows.append({
                **metadata,
                "tool": "kegg_api",
                "category": "KEGG_Pathway",
                "term_id": pathway_id,
                "term_name": self._pathway_names.get(pathway_id, pathway_id),
                "p_value": p_value,
                "q_value": np.nan,
                "correction": "BH-FDR",
                "background_gene_count": N,
                "total_genes_in_category": K,
                "query_gene_count": n,
                "overlap_count": k,
                "overlap_genes": ";".join(sorted(self._kegg_to_symbol.get(g, g) for g in overlap)),
                "query_symbols": ";".join(query_symbols),
                "query_kegg_ids": ";".join(sorted(query_ids)),
                "pathway_kegg_ids": ";".join(sorted(pathway_genes)),
            })
        if tested_rows:
            qvals = _bh_adjust([r["p_value"] for r in tested_rows])
            for row, q in zip(tested_rows, qvals):
                row["q_value"] = float(q)
            rows = [r for r in tested_rows if r["overlap_count"] > 0]
            rows.sort(key=lambda r: (r["q_value"], r["p_value"], r["term_id"]))
            rows = rows[:self.max_results]
        else:
            rows = []
        status = {**base_status, "status": "ok", "term_count": len(rows), "error": ""}
        return rows, status, mapping_rows

    def run_assignment_matrix(self, assignment: pd.DataFrame, dataset: str, fold: int, ablation: str, top_genes: int = None):
        if ablation not in self.variants:
            return
        top_n = max(1, int(top_genes or self.top_genes))
        if assignment.empty or assignment.shape[1] == 0:
            self.status_rows.append({"dataset": dataset, "fold": fold, "ablation": ablation, "module": "", "status": "empty_assignment", "error": "Assignment matrix is empty"})
            return
        columns = list(assignment.columns)
        max_modules = int(self.config.get("max_modules_per_fold", 0))
        if max_modules > 0:
            columns = columns[:max_modules]
        background_symbols = [str(g).strip() for g in assignment.index if str(g).strip() and str(g).lower() != "nan"]
        for module in columns:
            vals = pd.to_numeric(assignment[module], errors="coerce").to_numpy(float)
            order = np.argsort(np.nan_to_num(vals, nan=-np.inf))[::-1]
            genes = []
            for i in order:
                gene = str(assignment.index[i]).strip()
                if gene and gene.lower() != "nan" and gene not in genes:
                    genes.append(gene)
                if len(genes) >= top_n:
                    break
            metadata = {"dataset": str(dataset).upper(), "fold": int(fold), "ablation": str(ablation), "module": str(module), "top_n_requested": top_n}
            (self.input_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset)}_fold{fold}_{ablation}_{module}.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
            rows, status, maps = self.enrich_one_module(genes, background_symbols, metadata)
            self.result_rows.extend(rows)
            self.status_rows.append(status)
            self.mapping_rows.extend(maps)

    def write_outputs(self):
        result_columns = ["dataset", "fold", "ablation", "module", "top_n_requested", "tool", "category", "term_id", "term_name", "p_value", "q_value", "correction", "background_gene_count", "total_genes_in_category", "query_gene_count", "overlap_count", "overlap_genes", "query_symbols", "query_kegg_ids", "pathway_kegg_ids"]
        status_columns = ["dataset", "fold", "ablation", "module", "top_n_requested", "input_gene_count", "mapped_query_gene_count", "query_gene_count_in_background", "unmapped_query_gene_count", "background_symbol_count", "mapped_background_gene_count", "background_mode", "term_count", "status", "error"]
        mapping_columns = ["dataset", "fold", "ablation", "module", "top_n_requested", "submitted_symbol", "kegg_gene_id", "mapping_status"]
        result = pd.DataFrame(self.result_rows).reindex(columns=result_columns)
        status = pd.DataFrame(self.status_rows).reindex(columns=status_columns)
        mapping = pd.DataFrame(self.mapping_rows).reindex(columns=mapping_columns).drop_duplicates()
        result.to_csv(self.output_dir / "kegg_enrichment_per_module.csv", index=False)
        sig = result[pd.to_numeric(result["q_value"], errors="coerce") <= self.significance_qvalue] if not result.empty else result
        sig.to_csv(self.output_dir / "kegg_significant_pathways.csv", index=False)
        if not result.empty:
            summary = result.groupby(["dataset", "ablation", "module", "term_id", "term_name"], dropna=False).agg(
                folds_reported=("fold", "nunique"), min_p_value=("p_value", "min"), min_q_value=("q_value", "min"),
                mean_q_value=("q_value", "mean"), mean_overlap_count=("overlap_count", "mean")
            ).reset_index()
        else:
            summary = pd.DataFrame(columns=["dataset", "ablation", "module", "term_id", "term_name", "folds_reported", "min_p_value", "min_q_value", "mean_q_value", "mean_overlap_count"])
        summary.to_csv(self.output_dir / "kegg_pathway_fold_summary.csv", index=False)
        status.to_csv(self.output_dir / "kegg_query_status.csv", index=False)
        mapping.to_csv(self.output_dir / "kegg_gene_mapping.csv", index=False)
        meta = {
            "tool": "kegg_api", "endpoint": self.endpoint, "organism": self.organism,
            "method": "hypergeometric over-representation analysis with Benjamini-Hochberg FDR",
            "background_mode": self.background_mode, "pathway_count": len(self._pathway_to_genes),
            "gene_universe_size": len(self._universe), "modules_requested": len(self.status_rows),
            "successful_modules": sum(r.get("status") == "ok" for r in self.status_rows),
            "api_errors": sum(r.get("status") == "api_error" for r in self.status_rows),
            "note": "KEGG REST supplies annotation data; enrichment statistics are computed locally. Academic-use API, rate limited to <=3 requests/second."
        }
        (self.output_dir / "kegg_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"results": str(self.output_dir / "kegg_enrichment_per_module.csv"), "significant_terms": str(self.output_dir / "kegg_significant_pathways.csv"), "status": str(self.output_dir / "kegg_query_status.csv"), "mapping": str(self.output_dir / "kegg_gene_mapping.csv")}
