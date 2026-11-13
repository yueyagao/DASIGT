"""ToppFun API client and GMI-module enrichment automation.

Uses the official ToppGene Suite endpoints:
  POST /API/lookup  -- symbols to NCBI Entrez IDs
  POST /API/enrich  -- functional enrichment for Entrez IDs

All requests and normalized outputs are cached locally to support resuming long analyses.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import requests
except ImportError:  # keep non-enrichment analysis usable without the optional API dependency
    requests = None


DEFAULT_CATEGORIES = [
    "GeneOntologyBiologicalProcess",
    "GeneOntologyMolecularFunction",
    "GeneOntologyCellularComponent",
    "Pathway",
]


def _norm_symbol(value: Any) -> str:
    return str(value).strip().upper()


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class ToppFunRunner:
    """Run ToppFun enrichment for each module in a gene-assignment matrix."""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = dict(config or {})
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw_json"
        self.cache_dir = self.output_dir / "cache"
        self.input_dir = self.output_dir / "input_gene_lists"
        self.request_dir = self.output_dir / "requests"
        self.mapping_cache_path = self.cache_dir / "symbol_to_entrez.json"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.request_dir.mkdir(parents=True, exist_ok=True)

        self.endpoint = str(self.config.get("endpoint", "https://toppgene.cchmc.org/API")).rstrip("/")
        self.timeout = float(self.config.get("timeout_seconds", 120))
        self.retries = max(1, int(self.config.get("retries", 3)))
        self.request_interval = max(0.0, float(self.config.get("request_interval_seconds", 0.2)))
        self.lookup_batch_size = max(1, int(self.config.get("lookup_batch_size", 250)))
        self.use_cache = bool(self.config.get("use_cache", True))
        self.correction = str(self.config.get("correction", "Bonferroni"))
        self.pvalue_cutoff = float(self.config.get("pvalue", 0.05))
        self.min_genes = int(self.config.get("min_genes", 2))
        self.max_genes = int(self.config.get("max_genes", 1500))
        self.max_results = int(self.config.get("max_results", 100))
        self.background_mode = str(self.config.get("background", "selected_gene_universe"))
        self.top_genes = int(self.config.get("top_genes_per_module", 50))
        self.max_modules_per_fold = int(self.config.get("max_modules_per_fold", 0))
        self.variants = set(str(x) for x in self.config.get("variants", ["none"]))
        self.categories = self._build_categories(self.config.get("categories", DEFAULT_CATEGORIES))
        if requests is None:
            raise RuntimeError("ToppFun API integration requires the 'requests' package. Install it with: pip install requests")
        self.session = requests.Session()
        self.symbol_cache: Dict[str, Dict[str, Any]] = {}
        if self.use_cache and self.mapping_cache_path.exists():
            try:
                self.symbol_cache = json.loads(self.mapping_cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.symbol_cache = {}
        self.mapping_rows: List[Dict[str, Any]] = []
        self.result_rows: List[Dict[str, Any]] = []
        self.status_rows: List[Dict[str, Any]] = []

    def _build_categories(self, categories: Any) -> List[Dict[str, Any]]:
        if not categories:
            categories = DEFAULT_CATEGORIES
        built = []
        for item in categories:
            if isinstance(item, str):
                item = {"Type": item}
            if not isinstance(item, dict) or not item.get("Type"):
                continue
            built.append({
                "Type": str(item["Type"]),
                "PValue": float(item.get("PValue", self.config.get("pvalue", 0.05))),
                "MinGenes": int(item.get("MinGenes", self.config.get("min_genes", 2))),
                "MaxGenes": int(item.get("MaxGenes", self.config.get("max_genes", 1500))),
                "MaxResults": int(item.get("MaxResults", self.config.get("max_results", 100))),
                "Correction": str(item.get("Correction", self.config.get("correction", "Bonferroni"))),
            })
        return built

    def _post_json(self, endpoint: str, payload: Dict[str, Any], cache_key: Optional[str] = None) -> Dict[str, Any]:
        cache_path = self.raw_dir / f"{cache_key}.json" if cache_key else None
        if cache_key:
            request_path = self.request_dir / f"{cache_key}.request.json"
            request_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.use_cache and cache_path and cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        last_error = None
        for attempt in range(self.retries):
            try:
                if self.request_interval:
                    time.sleep(self.request_interval)
                started = time.time()
                print(f"[ToppFun API] POST /{endpoint.lstrip('/')} attempt={attempt + 1}/{self.retries}", flush=True)
                response = self.session.post(
                    f"{self.endpoint}/{endpoint.lstrip('/')}",
                    data=json.dumps(payload),
                    headers={"Content-Type": "text/json", "Accept": "application/json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                print(f"[ToppFun API] /{endpoint.lstrip('/')} HTTP {getattr(response, 'status_code', 'ok')} elapsed={time.time()-started:.2f}s", flush=True)
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError(f"Expected JSON object from ToppGene {endpoint}; got {type(data).__name__}")
                if data.get("Error") or data.get("error"):
                    raise RuntimeError(f"ToppGene returned an API error: {data.get('Error', data.get('error'))}")
                if self.use_cache and cache_path:
                    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                return data
            except Exception as exc:  # API/network errors are recorded per module; analysis continues.
                last_error = exc
                print(f"[ToppFun API] /{endpoint.lstrip('/')} failed: {type(exc).__name__}: {exc}", flush=True)
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"ToppGene API {endpoint} failed after {self.retries} attempts: {last_error}")

    def map_symbols(self, symbols: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
        """Map symbols to Entrez IDs; return records keyed by normalized submitted symbol."""
        normalized = list(dict.fromkeys(_norm_symbol(s) for s in symbols if str(s).strip()))
        missing_cache = [s for s in normalized if (not self.use_cache) or s not in self.symbol_cache]
        for start in range(0, len(missing_cache), self.lookup_batch_size):
            chunk = missing_cache[start:start + self.lookup_batch_size]
            payload = {"Symbols": chunk}
            cache_key = "lookup_" + _stable_hash(payload)
            try:
                data = self._post_json("lookup", payload, cache_key)
                returned = data.get("Genes", data.get("genes", [])) or []
                found = set()
                for gene in returned:
                    submitted = _norm_symbol(gene.get("Submitted", gene.get("submittedSymbol", "")))
                    official = str(gene.get("OfficialSymbol", gene.get("officialSymbol", ""))).strip()
                    entrez = gene.get("Entrez", gene.get("geneId", gene.get("entrez")))
                    try:
                        entrez_int = int(entrez) if entrez not in (None, "") else None
                    except (TypeError, ValueError):
                        entrez_int = None
                    if submitted:
                        self.symbol_cache[submitted] = {"entrez": entrez_int, "official_symbol": official or None,
                                                        "status": "mapped" if entrez_int is not None else "unmapped"}
                        found.add(submitted)
                    # The official symbol alias is useful when expression identifiers are mixed.
                    if official and entrez_int is not None:
                        self.symbol_cache.setdefault(_norm_symbol(official), {"entrez": entrez_int,
                            "official_symbol": official, "status": "mapped_alias"})
                missing_returned = data.get("Missing", data.get("missing", [])) or []
                missing_names = {_norm_symbol(x.get("Submitted", x.get("symbol", x)) if isinstance(x, dict) else x)
                                 for x in missing_returned}
                for sym in chunk:
                    if sym not in self.symbol_cache:
                        self.symbol_cache[sym] = {"entrez": None, "official_symbol": None,
                                                  "status": "unmapped" if (sym in missing_names or sym not in found) else "unmapped"}
            except Exception as exc:
                # Report the real lookup failure without referencing the retry-loop
                # variable `attempt` (which belongs to _post_json, not this scope).
                # Do not cache transient API failures as permanent unmapped symbols.
                print(
                    f"[ToppFun][LOOKUP ERROR] symbols={len(chunk)} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
        if self.use_cache:
            self.mapping_cache_path.write_text(json.dumps(self.symbol_cache, indent=2, ensure_ascii=False), encoding="utf-8")
        return {s: self.symbol_cache.get(s, {"entrez": None, "official_symbol": None, "status": "lookup_error"})
                for s in normalized}

    def _correction_value(self, annotation: Dict[str, Any]) -> Any:
        correction = self.correction.lower()
        if correction == "fdr":
            return annotation.get("QValueFDRBH", annotation.get("QValueFDR_BH"))
        if correction == "bonferroni":
            return annotation.get("QValueBonferroni", annotation.get("qValueBonferroni"))
        return annotation.get("PValue", annotation.get("pValue"))

    def enrich_one_module(
        self,
        gene_symbols: Sequence[Any],
        background_symbols: Sequence[Any],
        metadata: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        query_symbols = list(dict.fromkeys(str(g).strip() for g in gene_symbols if str(g).strip()))
        bg_symbols = list(dict.fromkeys(str(g).strip() for g in background_symbols if str(g).strip()))
        query_map = self.map_symbols(query_symbols)
        bg_map = self.map_symbols(bg_symbols) if self.background_mode == "selected_gene_universe" else {}
        query_ids = sorted({int(v["entrez"]) for v in query_map.values() if v.get("entrez") is not None})
        bg_ids = sorted({int(v["entrez"]) for v in bg_map.values() if v.get("entrez") is not None})
        mapping_rows = []
        for sym in query_symbols:
            rec = query_map.get(_norm_symbol(sym), {})
            mapping_rows.append({**metadata, "submitted_symbol": sym, "official_symbol": rec.get("official_symbol"),
                                 "entrez_id": rec.get("entrez"), "mapping_status": rec.get("status", "unmapped")})

        base_status = {
            **metadata,
            "input_gene_count": len(query_symbols),
            "mapped_query_gene_count": len(query_ids),
            "unmapped_query_gene_count": max(0, len(query_symbols) - len(query_ids)),
            "background_symbol_count": len(bg_symbols) if self.background_mode == "selected_gene_universe" else 0,
            "mapped_background_gene_count": len(bg_ids),
            "background_mode": self.background_mode,
            "term_count": 0,
            "status": "pending",
            "error": "",
        }
        lookup_errors = any(v.get("status") == "lookup_error" for v in query_map.values())
        if len(query_ids) < self.min_genes:
            state = "api_error" if lookup_errors else "skipped_too_few_mapped_genes"
            message = "Gene symbol lookup API failed" if lookup_errors else f"Only {len(query_ids)} query genes mapped; min_genes={self.min_genes}"
            return [], {**base_status, "status": state, "error": message}, mapping_rows
        if self.background_mode == "selected_gene_universe" and bg_symbols and not bg_ids:
            bg_lookup_errors = any(v.get("status") == "lookup_error" for v in bg_map.values())
            if bg_lookup_errors:
                return [], {**base_status, "status": "api_error", "error": "Background gene symbol lookup API failed"}, mapping_rows
        background_ids = bg_ids if self.background_mode == "selected_gene_universe" else []
        if background_ids and not set(query_ids).issubset(set(background_ids)):
            background_ids = sorted(set(background_ids).union(query_ids))
        payload = {"Genes": query_ids, "Background": background_ids, "Categories": self.categories}
        cache_key = "enrich_" + _stable_hash(payload)
        try:
            data = self._post_json("enrich", payload, cache_key)
            annotations = data.get("Annotations", data.get("annotations", data.get("results", []))) or []
            rows = []
            for ann in annotations:
                genes = ann.get("Genes", ann.get("genes", [])) or []
                overlap_symbols = []
                for g in genes:
                    if isinstance(g, dict):
                        sym = g.get("Symbol", g.get("symbol", g.get("OfficialSymbol")))
                        if sym:
                            overlap_symbols.append(str(sym))
                    elif g:
                        overlap_symbols.append(str(g))
                try:
                    p_value = float(ann.get("PValue", ann.get("pValue", np.nan)))
                except (TypeError, ValueError):
                    p_value = np.nan
                q_value = self._correction_value(ann)
                try:
                    q_value = float(q_value) if q_value not in (None, "") else np.nan
                except (TypeError, ValueError):
                    q_value = np.nan
                rows.append({
                    **metadata,
                    "category": ann.get("Category", ann.get("category")),
                    "term_id": ann.get("ID", ann.get("id")),
                    "term_name": ann.get("Name", ann.get("name")),
                    "p_value": p_value,
                    "q_value": q_value,
                    "correction": self.correction,
                    "total_genes_in_category": ann.get("TotalGenes", ann.get("totalGenes")),
                    "genes_in_term": ann.get("GenesInTerm", ann.get("genesInTerm")),
                    "genes_in_query": ann.get("GenesInQuery", ann.get("genesInQuery")),
                    "overlap_count": ann.get("GenesInTermInQuery", ann.get("genesInTermInQuery")),
                    "overlap_genes": ";".join(sorted(set(overlap_symbols))),
                    "query_gene_count": len(query_ids),
                    "background_gene_count": len(background_ids),
                    "query_symbols": ";".join(query_symbols),
                    "query_entrez_ids": ";".join(map(str, query_ids)),
                })
            return rows, {**base_status, "status": "ok", "term_count": len(rows), "error": ""}, mapping_rows
        except Exception as exc:
            return [], {**base_status, "status": "api_error", "error": str(exc)}, mapping_rows

    def run_assignment_matrix(
        self,
        assignment: pd.DataFrame,
        dataset: str,
        fold: int,
        ablation: str,
        top_genes: Optional[int] = None,
    ) -> None:
        if ablation not in self.variants:
            return
        top_n = max(1, int(top_genes or self.top_genes))
        if assignment.empty or assignment.shape[1] == 0:
            self.status_rows.append({"dataset": dataset, "fold": fold, "ablation": ablation,
                                     "module": "", "status": "empty_assignment", "error": "Assignment matrix is empty"})
            return
        module_columns = list(assignment.columns)
        if self.max_modules_per_fold > 0:
            module_columns = module_columns[:self.max_modules_per_fold]
        background_symbols = [str(g).strip() for g in assignment.index if str(g).strip()]
        for module_col in module_columns:
            values = pd.to_numeric(assignment[module_col], errors="coerce").to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=-np.inf)
            order = np.argsort(values)[::-1]
            order = [i for i in order if np.isfinite(values[i])][:top_n]
            genes = [str(assignment.index[i]).strip() for i in order if str(assignment.index[i]).strip()]
            metadata = {"dataset": dataset.upper(), "fold": int(fold), "ablation": ablation,
                        "module": str(module_col), "top_n_requested": top_n}
            input_path = self.input_dir / f"{_safe_name(dataset)}_fold{fold}_{_safe_name(ablation)}_{_safe_name(module_col)}.txt"
            input_path.write_text("\n".join(genes) + ("\n" if genes else ""), encoding="utf-8")
            rows, status, mappings = self.enrich_one_module(genes, background_symbols, metadata)
            self.result_rows.extend(rows)
            self.status_rows.append(status)
            self.mapping_rows.extend(mappings)

    def write_outputs(self) -> Dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_cols = ["dataset", "fold", "ablation", "module", "top_n_requested", "category", "term_id", "term_name",
                       "p_value", "q_value", "correction", "total_genes_in_category", "genes_in_term", "genes_in_query",
                       "overlap_count", "overlap_genes", "query_gene_count", "background_gene_count", "query_symbols", "query_entrez_ids"]
        status_cols = ["dataset", "fold", "ablation", "module", "top_n_requested", "input_gene_count", "mapped_query_gene_count",
                       "unmapped_query_gene_count", "background_symbol_count", "mapped_background_gene_count", "background_mode",
                       "term_count", "status", "error"]
        mapping_cols = ["dataset", "fold", "ablation", "module", "top_n_requested", "submitted_symbol", "official_symbol", "entrez_id", "mapping_status"]
        results = pd.DataFrame(self.result_rows)
        status = pd.DataFrame(self.status_rows)
        mapping = pd.DataFrame(self.mapping_rows)
        if not results.empty:
            results = results.reindex(columns=result_cols)
            results.to_csv(self.output_dir / "toppfun_enrichment_per_module.csv", index=False)
            qcut = float(self.config.get("significance_qvalue", 0.05))
            sig = results[pd.to_numeric(results["q_value"], errors="coerce") <= qcut].copy()
            sig.to_csv(self.output_dir / "toppfun_significant_terms.csv", index=False)
            group_cols = ["dataset", "ablation", "module", "category", "term_id", "term_name"]
            agg = results.groupby(group_cols, dropna=False).agg(
                folds_reported=("fold", "nunique"), min_p_value=("p_value", "min"), min_q_value=("q_value", "min"),
                mean_q_value=("q_value", "mean"), mean_overlap_count=("overlap_count", "mean"),
            ).reset_index()
            agg.to_csv(self.output_dir / "toppfun_term_fold_summary.csv", index=False)
        else:
            pd.DataFrame(columns=result_cols).to_csv(self.output_dir / "toppfun_enrichment_per_module.csv", index=False)
            pd.DataFrame(columns=result_cols).to_csv(self.output_dir / "toppfun_significant_terms.csv", index=False)
            pd.DataFrame(columns=["dataset", "ablation", "module", "category", "term_id", "term_name", "folds_reported",
                                  "min_p_value", "min_q_value", "mean_q_value", "mean_overlap_count"]).to_csv(
                                      self.output_dir / "toppfun_term_fold_summary.csv", index=False)
        if not status.empty:
            status = status.reindex(columns=status_cols)
        else:
            status = pd.DataFrame(columns=status_cols)
        status.to_csv(self.output_dir / "toppfun_query_status.csv", index=False)
        if not mapping.empty:
            mapping = mapping.reindex(columns=mapping_cols).drop_duplicates()
        else:
            mapping = pd.DataFrame(columns=mapping_cols)
        mapping.to_csv(self.output_dir / "toppfun_gene_mapping.csv", index=False)
        manifest = {
            "service": "ToppGene Suite API",
            "endpoint": self.endpoint,
            "categories": self.categories,
            "correction": self.correction,
            "pvalue_cutoff": self.pvalue_cutoff,
            "background_mode": self.background_mode,
            "top_genes_per_module": self.top_genes,
            "variants": sorted(self.variants),
            "query_count": len(self.status_rows),
            "successful_queries": sum(x.get("status") == "ok" for x in self.status_rows),
            "api_errors": sum(x.get("status") == "api_error" for x in self.status_rows),
            "note": "ToppFun is a remote API; query gene lists are sent to the ToppGene server. Raw API JSON and cache are retained for reproducibility.",
        }
        (self.output_dir / "toppfun_run_metadata.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "results": str(self.output_dir / "toppfun_enrichment_per_module.csv"),
            "significant_terms": str(self.output_dir / "toppfun_significant_terms.csv"),
            "term_summary": str(self.output_dir / "toppfun_term_fold_summary.csv"),
            "status": str(self.output_dir / "toppfun_query_status.csv"),
            "mapping": str(self.output_dir / "toppfun_gene_mapping.csv"),
            "metadata": str(self.output_dir / "toppfun_run_metadata.json"),
        }
