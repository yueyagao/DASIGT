import tempfile, unittest, sys, types
from pathlib import Path
import pandas as pd, yaml
# The overlay is tested standalone; the target DASIGT project already provides config.load_config.
if 'dasigt.config' not in sys.modules:
    _config_mod = types.ModuleType('dasigt.config')
    _config_mod.load_config = lambda path: yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    sys.modules['dasigt.config'] = _config_mod
from dasigt.toppfun_transfer import prepare_bundle

class ToolSelectionTest(unittest.TestCase):
    def test_prepare_bundle_pins_kegg_tool_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'configs').mkdir()
            cfg={
              'project': {'datasets':['BRCA'],'output_root':str(root/'outputs'),'summary_analysis':str(root/'summary/analysis')},
              'training': {'folds':[0]},
              'analysis': {'enrichment_tool':'kegg','pathway_top_genes':2,'toppfun':{
                'enabled':False,'top_genes_per_module':2,'variants':['none'],'max_modules_per_fold':1,
                'background':'selected_gene_universe','categories':['GeneOntologyBiologicalProcess']}}
            }
            cp=root/'configs/config.yaml'; cp.write_text(yaml.safe_dump(cfg),encoding='utf-8')
            out=root/'outputs/BRCA/fold_0/none'; out.mkdir(parents=True)
            pd.DataFrame({'module_aligned_0':[0.9,0.2,0.1]},index=['A','B','C']).to_csv(out/'gene_module_assignments.csv')
            archive=prepare_bundle(str(cp))
            bundle=root/'summary/analysis/toppfun_transfer_bundle'
            import json
            ec=json.loads((bundle/'enrichment_config.json').read_text())
            self.assertEqual(ec['tool'],'kegg_api')
            self.assertTrue(archive.exists())
            genes=(bundle/'input_gene_lists/BRCA_fold0_none_module_aligned_0.txt').read_text().splitlines()
            self.assertEqual(genes,['A','B'])

if __name__=='__main__': unittest.main()
