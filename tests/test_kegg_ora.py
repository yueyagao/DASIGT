import tempfile
import unittest
from pathlib import Path

from dasigt.kegg import KEGGEnrichmentRunner


class KEGGORATest(unittest.TestCase):
    def test_hypergeometric_and_fdr_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            r = KEGGEnrichmentRunner.__new__(KEGGEnrichmentRunner)
            r.output_dir=Path(td); r.cache_dir=Path(td)/'cache'; r.raw_dir=Path(td)/'raw_json'; r.request_dir=Path(td)/'requests'; r.input_dir=Path(td)/'input_gene_lists'
            for p in (r.cache_dir,r.raw_dir,r.request_dir,r.input_dir): p.mkdir(parents=True,exist_ok=True)
            r.endpoint='https://rest.kegg.jp'; r.organism='hsa'; r.background_mode='all_category_genes'; r.top_genes=3; r.min_genes=2; r.max_results=20; r.significance_qvalue=.05; r.variants={'none'}
            r._symbol_to_kegg={'A':'hsa:1','B':'hsa:2','C':'hsa:3','D':'hsa:4'}
            r._kegg_to_symbol={'hsa:1':'A','hsa:2':'B','hsa:3':'C','hsa:4':'D'}
            r._pathway_to_genes={'hsa00010':{'hsa:1','hsa:2','hsa:3'},'hsa00020':{'hsa:3','hsa:4'}}
            r._pathway_names={'hsa00010':'Glycolysis','hsa00020':'Other pathway'}
            r._universe={'hsa:1','hsa:2','hsa:3','hsa:4'}
            r.result_rows=[]; r.status_rows=[]; r.mapping_rows=[]
            rows,status,mapping=r.enrich_one_module(['A','B','C'],[],{'dataset':'BRCA','fold':0,'ablation':'none','module':'module_aligned_0','top_n_requested':3})
            self.assertEqual(status['status'],'ok')
            self.assertEqual(rows[0]['term_id'],'hsa00010')
            self.assertAlmostEqual(rows[0]['p_value'], 0.25)
            self.assertEqual(status['mapped_query_gene_count'], 3)
            self.assertEqual(status['query_gene_count_in_background'], 3)
            r.result_rows.extend(rows); r.status_rows.append(status); r.mapping_rows.extend(mapping)
            paths=r.write_outputs()
            self.assertTrue(Path(paths['results']).exists())
            self.assertTrue((Path(td)/'kegg_run_metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
