#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python - <<'PY'
from pathlib import Path
import tempfile, yaml, numpy as np, pandas as pd, torch
from dasigt.data_prep import run_data_prep
from dasigt.dataset import prepare_fold
from dasigt.train import run_training
from dasigt.analysis import run_analysis
root=Path(tempfile.mkdtemp(prefix='dasigt_toy_'))
raw=root/'raw'
for d in ['image_path/TCGA_BRCA/pt_files','sent_path/TCGA_BRCA/pt_files','mrna_path','clinical_path','gene2vec_path']:(raw/d).mkdir(parents=True,exist_ok=True)
pids=[f'TCGA-AA-{i:04d}' for i in range(1,9)]
genes=[f'G{i}' for i in range(12)]
rng=np.random.default_rng(1)
with open(raw/'gene2vec_path/gene2vec.txt','w') as f:
 for g in genes:f.write(g+' '+' '.join(map(str,rng.normal(size=200).astype(float)))+'\n')
expr=pd.DataFrame(rng.normal(size=(12,8)),index=genes,columns=[p+'-01A-01R' for p in pids]);expr.to_csv(raw/'mrna_path/BRCA.csv')
clin=pd.DataFrame({'case_id':pids,'vital_status':[1,1,1,1,0,0,1,0], 'days_to_death':[100,200,300,400,np.nan,np.nan,700,np.nan], 'days_to_last_followup':[np.nan,np.nan,np.nan,np.nan,500,600,np.nan,800]})
clin.to_csv(raw/'clinical_path/BRCA.csv',index=False)
for pid in pids:
 torch.save(torch.randn(4,512),raw/'image_path/TCGA_BRCA/pt_files'/f'{pid}-01Z-00-DX1.pt')
 torch.save(torch.randn(3,512),raw/'sent_path/TCGA_BRCA/pt_files'/f'{pid}.pt')
splitroot=root/'splits/tcga_brca';splitroot.mkdir(parents=True)
pd.DataFrame({'train':pids[:5]+[None]*3,'val':pids[5:7]+[None]*6,'test':pids[5:7]+[None]*6}).to_csv(splitroot/'splits_0.csv',index=False)
cfg={'project':{'name':'dasigt','seed':7,'datasets':['BRCA'],'output_root':str(root/'outputs'),'real_use_data_root':str(root/'real_use_data'),'summary_data_process':str(root/'summary_table/data_process'),'summary_analysis':str(root/'summary_table/analysis'),'device':'cpu','num_workers':0},
'paths':{'raw_root':str(raw),'image_dir':'image_path','text_dir':'sent_path','gene_expression_dir':'mrna_path','gene2vec_file':'gene2vec_path/gene2vec.txt','clinical_dir':'clinical_path','split_root':str(root/'splits'),'split_pattern':'tcga_{dataset_lower}/splits_{fold}.csv','gene_id_map_file':''},
'features':{'image_dim':512,'text_dim':512,'gene2vec_dim':200,'feature_pooling':'mean','max_image_tokens':8,'max_text_tokens':8,'gene_id_separator':'|','gene_selection':'variance','top_gene_num':8,'gene_cluster_num':3,'gene_temperature':0.5,'gene_center_init':'kmeans','gene_kmeans_n_init':2,'gene_kmeans_max_iter':30,'gene_power_steps':2,'gene_epsilon':1e-8,'gene_center_svd':False},
'survival':{'time_bins':4,'time_unit':'days','clinical_id_col':'case_id','event_col':'vital_status','event_positive_values':[1,'1','dead','Dead','DEAD','deceased','Deceased'],'time_col':'','death_time_col':'days_to_death','censor_time_col':'days_to_last_followup','fit_cutpoints_on_train_events_only':True},
'model':{'name':'dasigt','hidden_dim':512,'attention_heads':8,'attention_dropout':0.0,'dropout':0.1,'num_experts':4,'routing_temperature':1.0,'routing_gumbel':True,'routing_straight_through':True,'routing_eval':'argmax','lambda_gene':1.0,'lambda_gmi_compact':0.01,'lambda_gmi_balance':0.01,'ablation':'none'},
 'training':{'folds':[0],'max_epochs':1,'patience':1,'batch_size':1,'gradient_accumulation_steps':1,'optimizer':'adamw','learning_rate':0.0001,'weight_decay':0.00001,'grad_clip_norm':5.0,'use_amp':False,'monitor':'val_cindex','save_predictions':True,'allow_val_test_overlap':True},
 'ablation':{'variants':['none']},'analysis':{'export_gene_module_weights':True,'export_route_logits':True,'export_patient_predictions':True,'route_entropy':True,'pathway_top_genes':3,'enrichment_tool':'not_configured','enrichment_background':'selected_gene_universe'}}
conf=root/'config.yaml'; conf.write_text(yaml.safe_dump(cfg,sort_keys=False))
run_data_prep(conf)
fold=prepare_fold(root/'real_use_data',splitroot/'splits_0.csv',cfg,'BRCA',0,root/'outputs')
assert len(fold['train'])==5 and fold['prior'].shape==(8,200) and fold['train_expression'].shape==(5,8)
print('Toy preprocessing/fold setup passed:',root)
if __import__('os').environ.get('DASIGT_FULL_TOY')=='1':
 run_training(conf,['BRCA'],[0],None)
 run_analysis(conf)
 print('Toy one-epoch training and analysis passed:',root)
PY
