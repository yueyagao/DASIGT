"""Quick synthetic checks that do not require access to the full TCGA feature store."""
import torch
from .model import DASIGT
from .survival import discrete_survival_nll


def run_smoke_test():
    torch.manual_seed(3)
    cfg={"model":{"hidden_dim":512,"attention_heads":8,"attention_dropout":0.0,"dropout":0.1,"routing_temperature":1.0,
                  "routing_gumbel":True,"routing_straight_through":True,"num_experts":4,"lambda_gene":1.0,"lambda_gmi_compact":0.01,"lambda_gmi_balance":0.01,"ablation":"none"},
         "features":{"image_dim":512,"text_dim":512,"gene_cluster_num":4,"gene_temperature":0.5,"gene_power_steps":2,"gene_epsilon":1e-8,"gene2vec_dim":200},
         "survival":{"time_bins":4}}
    prior=torch.randn(20,200); train_expr=torch.randn(8,20)
    model=DASIGT(prior,train_expr,cfg)

    # True-batch check with different valid token lengths and explicit masks.
    image=torch.randn(3,7,512); text=torch.randn(3,5,512); gene=torch.randn(3,20)
    image_mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
    text_mask=torch.tensor([[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    model.train()
    h,s,r,gl,d=model(image,text,gene,image_mask=image_mask,text_mask=text_mask,return_details=True)
    loss=discrete_survival_nll(h,torch.tensor([1,2,3]),torch.tensor([1,0,1]))+gl
    loss.backward()
    assert h.shape==(3,4) and s.shape==(3,4)
    assert model.gmi.prior_shift.weight.grad is not None
    assert torch.isfinite(model.gmi.prior_shift.weight.grad).all()
    assert model.gmi.centroids.grad is not None and torch.isfinite(model.gmi.centroids.grad).all()

    # Backward-compatible single-patient interface still works.
    model.eval()
    with torch.no_grad():
        h2,s2,r2,gl2,d2=model(image[0,:4],text[0,:2],gene[0],return_details=True)
    assert h2.shape==(1,4) and torch.isfinite(h2).all() and torch.isfinite(s2).all()
    print("DASIGT true-batch and single-patient forward/backward smoke tests passed.")

if __name__=="__main__": run_smoke_test()
