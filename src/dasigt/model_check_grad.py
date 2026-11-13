"""DASIGT model: soft GMI, differentiable Power Iteration, and two-stage routing."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def gumbel_softmax_st(logits, temperature=1.0, training=True, use_gumbel=True, hard=True):
    if training and use_gumbel:
        u=torch.rand_like(logits).clamp_(1e-8,1-1e-8)
        g=-torch.log(-torch.log(u))
        y=F.softmax((logits+g)/max(temperature,1e-5),dim=-1)
    else:
        y=F.softmax(logits/max(temperature,1e-5),dim=-1)
    if not hard: return y
    idx=y.argmax(dim=-1,keepdim=True)
    y_hard=torch.zeros_like(y).scatter_(-1,idx,1.0)
    return y_hard-y.detach()+y


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.norm=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*2,dim))
        self.norm2=nn.LayerNorm(dim)
    def forward(self,q,k=None,v=None):
        if k is None: k=q
        if v is None: v=k
        z,_=self.attn(q,k,v,need_weights=False)
        q=self.norm(q+z); return self.norm2(q+self.ff(q))


class GeneModuleIdentification(nn.Module):
    """Fold-specific GMI. Prior projection is scalar; memberships and PI remain differentiable."""
    def __init__(self, gene_prior, train_expression, n_modules=50, temperature=0.5, power_steps=8, eps=1e-8,
                 lambda_compact=0.01, lambda_balance=0.01, use_prior=True, use_gene_loss=True, hidden_dim=512,
                 kmeans_n_init=10, kmeans_max_iter=300, kmeans_random_state=7):
        super().__init__()
        prior=torch.as_tensor(gene_prior,dtype=torch.float32)
        expr=torch.as_tensor(train_expression,dtype=torch.float32)
        if prior.ndim!=2 or expr.ndim!=2 or expr.shape[1]!=prior.shape[0]:
            raise ValueError(f"GMI shape mismatch: prior={tuple(prior.shape)} train_expression={tuple(expr.shape)}")
        self.n_genes=prior.shape[0]; self.n_modules=min(int(n_modules),self.n_genes); self.temperature=float(temperature)
        self.power_steps=int(power_steps); self.eps=float(eps); self.lambda_compact=lambda_compact; self.lambda_balance=lambda_balance
        self.use_prior=use_prior; self.use_gene_loss=use_gene_loss
        self.register_buffer("gene_prior",prior)
        self.register_buffer("train_expression",expr)
        self.prior_shift=nn.Linear(prior.shape[1],1,bias=False)
        # Initialize centers on the layer's current device. During DASIGT construction,
        # the input buffers may already be on CUDA while nn.Linear parameters are still
        # on CPU (the enclosing model is moved with .to(device) only after __init__).
        # This initialization is non-differentiable, so temporary detached copies are safe.
        init_device = self.prior_shift.weight.device
        init_dtype = self.prior_shift.weight.dtype
        prior_init = prior.detach().to(device=init_device, dtype=init_dtype)
        expr_init = expr.detach().to(device=init_device, dtype=init_dtype)
        with torch.no_grad():
            profiles = (expr_init + self.prior_shift(prior_init).T).T.cpu().numpy()
        try:
            from sklearn.cluster import KMeans
            km=KMeans(n_clusters=self.n_modules,n_init=int(kmeans_n_init),max_iter=int(kmeans_max_iter),random_state=int(kmeans_random_state)).fit(profiles)
            centers=torch.tensor(km.cluster_centers_,dtype=torch.float32)
        except Exception:
            ids=torch.linspace(0,max(0,profiles.shape[0]-1),self.n_modules).long().numpy()
            centers=torch.tensor(profiles[ids],dtype=torch.float32)
        self.centroids=nn.Parameter(centers)
        self.gene_encoder=nn.Sequential(nn.Linear(self.n_modules,hidden_dim),nn.GELU(),nn.Dropout(0.1),nn.Linear(hidden_dim,hidden_dim))
        self._cached_loadings=None

    def forward(self, patient_expression, return_details=False):
        prior=self.gene_prior
        shift=self.prior_shift(prior).squeeze(-1) if self.use_prior else torch.zeros(self.n_genes,device=prior.device)
        train_x=self.train_expression+shift.unsqueeze(0)
        profiles=train_x.T  # genes x training patients
        # Soft assignment over gene-profile distance, differentiable w.r.t. centers and prior shift.
        dist=torch.cdist(profiles.unsqueeze(0),self.centroids.unsqueeze(0)).squeeze(0).pow(2)
        assign=F.softmax(-dist/max(self.temperature,1e-5),dim=-1)  # genes x modules
        # Debug hook: retain the non-leaf gradient so training can verify that
        # the soft gene-module assignment participates in backpropagation.
        if self.training and assign.requires_grad:
            assign.retain_grad()
        weighted=torch.sqrt(assign.clamp_min(self.eps))  # genes x modules
        # Vectorized Power Iteration: all modules are updated in two batched matrix multiplications.
        v=torch.ones((self.n_genes,self.n_modules),device=train_x.device,dtype=train_x.dtype)
        v=v/(v.norm(dim=0,keepdim=True)+self.eps)
        # Keep selected Power Iteration intermediates for optional graph-connectivity diagnostics.
        pi_first_u=pi_first_v=pi_last_u=pi_last_v=None
        for step in range(self.power_steps):
            u=train_x@(weighted*v)
            u=u/(u.norm(dim=0,keepdim=True)+self.eps)
            v=(train_x.T@u)*weighted
            v=v/(v.norm(dim=0,keepdim=True)+self.eps)
            if step == 0:
                pi_first_u,pi_first_v=u,v
            pi_last_u,pi_last_v=u,v
        # Deterministic sign convention, independently per module.
        pivots=v.abs().argmax(dim=0,keepdim=True)
        pivot_values=v.gather(0,pivots)
        signs=torch.where(pivot_values<0,-torch.ones_like(pivot_values),torch.ones_like(pivot_values))
        v=v*signs
        loadings=weighted*v
        test_x=patient_expression+shift
        scores=(test_x.unsqueeze(0)@loadings).squeeze(0).unsqueeze(0)
        h_gene=self.gene_encoder(scores)
        if self.use_gene_loss:
            # Compactness on gene profiles and balanced use of modules; this term is differentiable.
            recon=assign@self.centroids
            compact=((profiles-recon).pow(2).mean())
            usage=assign.mean(dim=0)
            target=torch.full_like(usage,1.0/self.n_modules)
            balance=(usage-target).pow(2).mean()
            gene_loss=self.lambda_compact*compact+self.lambda_balance*balance
        else: gene_loss=torch.zeros((),device=patient_expression.device)
        if return_details:
            return h_gene, gene_loss, {
                "assignment":assign,
                "module_scores":scores.squeeze(0),
                "loadings":loadings,
                "gene_shift":shift,
                # These tensors are diagnostic taps into the differentiable Power Iteration graph.
                "pi_first_u":pi_first_u,
                "pi_first_v":pi_first_v,
                "pi_last_u":pi_last_u,
                "pi_last_v":pi_last_v,
            }
        return h_gene,gene_loss


class DASIGT(nn.Module):
    """Original ExMP-style multimodal backbone under the unified model name DASIGT."""
    ROUTES=((1,0,0),(1,1,0),(1,0,1),(1,1,1))  # image-anchored candidate routes
    def __init__(self, gene_prior, train_expression, config, ablation=None):
        super().__init__(); mc=config["model"]; fc=config["features"]
        self.dim=int(mc.get("hidden_dim",512)); self.n_bins=int(config["survival"]["time_bins"])
        self.ablation=ablation or mc.get("ablation","none"); self.routing_temp=float(mc.get("routing_temperature",1.0))
        self.use_gumbel=bool(mc.get("routing_gumbel",True)); self.straight_through=bool(mc.get("routing_straight_through",True)); self.use_gene_loss=self.ablation!="no_gene_loss"
        if int(mc.get("num_experts",4)) != 4: raise ValueError("DASIGT architecture follows the original four-expert design; model.num_experts must be 4.")
        self.image_encoder=nn.Sequential(nn.Linear(fc["image_dim"],self.dim),nn.GELU(),nn.Dropout(mc.get("dropout",0.25)),nn.Linear(self.dim,self.dim))
        self.text_encoder=nn.Sequential(nn.Linear(fc["text_dim"],self.dim),nn.GELU(),nn.Dropout(mc.get("dropout",0.25)),nn.Linear(self.dim,self.dim))
        self.gmi=GeneModuleIdentification(gene_prior,train_expression,fc["gene_cluster_num"],fc["gene_temperature"],fc["gene_power_steps"],fc["gene_epsilon"],mc.get("lambda_gmi_compact",0.01),mc.get("lambda_gmi_balance",0.01),use_prior=self.ablation!="no_gene_prior",use_gene_loss=self.use_gene_loss,hidden_dim=self.dim,kmeans_n_init=fc.get("gene_kmeans_n_init",10),kmeans_max_iter=fc.get("gene_kmeans_max_iter",300),kmeans_random_state=fc.get("gene_kmeans_random_state",7))
        self.route_mlp=nn.Sequential(nn.Linear(self.dim*3,self.dim),RMSNorm(self.dim),nn.GELU(),nn.Dropout(0.1))
        self.modality_gate=nn.Linear(self.dim,4); self.expert_gate=nn.Linear(self.dim,4)
        def attn(): return CrossAttention(self.dim,mc.get("attention_heads",8),mc.get("attention_dropout",0.1))
        # Keep the original ExMP parameterization: EX1/EX2 use sequential cross-attention,
        # EX3 uses additive aggregation, and EX4 uses joint self-attention.
        self.experts=nn.ModuleList([
            nn.ModuleDict({"self_i":attn(),"self_g":attn(),"self_r":attn(),"cross1":attn(),"cross2":attn()}),
            nn.ModuleDict({"self_i":attn(),"self_g":attn(),"self_r":attn(),"cross1":attn(),"cross2":attn()}),
            nn.ModuleDict({"self_i":attn(),"self_g":attn(),"self_r":attn()}),
            nn.ModuleDict({"self_i":attn(),"joint":attn()})
        ])
        self.expert_proj=nn.ModuleList([nn.Sequential(nn.Linear(self.dim,self.dim),nn.GELU(),nn.Dropout(0.1)) for _ in range(4)])
        self.classifier=nn.Sequential(nn.LayerNorm(self.dim),nn.Dropout(mc.get("dropout",0.25)),nn.Linear(self.dim,self.n_bins))

    def _route(self, route_idx, image, gene, text, expert_idx):
        e=self.experts[expert_idx]
        # Individual expert interaction patterns inspired by the original ExMP experts.
        if route_idx==0:  # image only
            z=e["self_i"](image); return self.expert_proj[expert_idx](z.mean(dim=1))
        if route_idx==1:  # image + gene
            if gene.shape[1] == 0:
                z=e["self_i"](image); return self.expert_proj[expert_idx](z.mean(dim=1))
            if expert_idx==3:
                z=e["joint"](torch.cat([image,gene],dim=1)); z=z[:,:image.shape[1]]
                return self.expert_proj[expert_idx](z.mean(dim=1))
            gi=e["self_g"](gene)
            if expert_idx==0: z=e["cross1"](image,gi,gi)
            elif expert_idx==1: z=e["cross1"](image,gi,gi); z=e["cross2"](z,gi,gi)
            elif expert_idx==2: z=e["self_i"](image)+gi.mean(dim=1,keepdim=True)
            else: z=e["joint"](torch.cat([image,gi],dim=1)); z=z[:,:image.shape[1]]
            return self.expert_proj[expert_idx](z.mean(dim=1))
        if route_idx==2:  # image + report
            if text.shape[1] == 0:
                z=e["self_i"](image); return self.expert_proj[expert_idx](z.mean(dim=1))
            if expert_idx==3:
                z=e["joint"](torch.cat([image,text],dim=1)); z=z[:,:image.shape[1]]
                return self.expert_proj[expert_idx](z.mean(dim=1))
            rt=e["self_r"](text)
            if expert_idx==0: z=e["cross1"](image,rt,rt)
            elif expert_idx==1: z=e["cross1"](image,rt,rt); z=e["cross2"](z,rt,rt)
            elif expert_idx==2: z=e["self_i"](image)+rt.mean(dim=1,keepdim=True)
            else: z=e["joint"](torch.cat([image,rt],dim=1)); z=z[:,:image.shape[1]]
            return self.expert_proj[expert_idx](z.mean(dim=1))
        # image + gene + report (ablation may disable either auxiliary modality)
        if gene.shape[1] == 0 and text.shape[1] == 0:
            z=e["self_i"](image); return self.expert_proj[expert_idx](z.mean(dim=1))
        if gene.shape[1] == 0:
            rt=e["self_r"](text)
            z=e["cross1"](image,rt,rt) if expert_idx in (0,1) else e["self_i"](image)+rt.mean(dim=1,keepdim=True)
            return self.expert_proj[expert_idx](z.mean(dim=1))
        if text.shape[1] == 0:
            gi=e["self_g"](gene)
            z=e["cross1"](image,gi,gi) if expert_idx in (0,1) else e["self_i"](image)+gi.mean(dim=1,keepdim=True)
            return self.expert_proj[expert_idx](z.mean(dim=1))
        if expert_idx==3:
            z=e["joint"](torch.cat([image,gene,text],dim=1)); z=z[:,:image.shape[1]]
            return self.expert_proj[expert_idx](z.mean(dim=1))
        gi=e["self_g"](gene); rt=e["self_r"](text)
        if expert_idx==0:
            z=e["cross1"](image,gi,gi); z=e["cross2"](z,rt,rt)
        elif expert_idx==1:
            z=e["cross1"](image,rt,rt); z=e["cross2"](z,gi,gi)
        elif expert_idx==2:
            z=e["self_i"](image)+gi.mean(dim=1,keepdim=True)+rt.mean(dim=1,keepdim=True)
        else:
            z=e["joint"](torch.cat([image,gi,rt],dim=1)); z=z[:,:image.shape[1]]
        return self.expert_proj[expert_idx](z.mean(dim=1))

    def forward(self, image_features, text_features, patient_expression, return_details=False):
        image=self.image_encoder(image_features.unsqueeze(0) if image_features.ndim==2 else image_features)
        text=self.text_encoder(text_features.unsqueeze(0) if text_features.ndim==2 else text_features)
        gene_vec,gene_loss,gene_details=self.gmi(patient_expression,return_details=True)
        gene=gene_vec.unsqueeze(1)
        image_pool=image.mean(dim=1); text_pool=text.mean(dim=1); gene_pool=gene.mean(dim=1)
        gate_in=self.route_mlp(torch.cat([image_pool,gene_pool,text_pool],dim=-1))
        modality_logits=self.modality_gate(gate_in); expert_logits=self.expert_gate(gate_in)
        # Debug hook: retain gradients on the two routing-logit tensors.
        if self.training:
            if modality_logits.requires_grad:
                modality_logits.retain_grad()
            if expert_logits.requires_grad:
                expert_logits.retain_grad()
        hard_route=(not self.training) or self.straight_through
        if self.ablation in {"fixed_trifusion","no_modality_gate"}:
            route_weights=torch.zeros_like(modality_logits); route_weights[:,3]=1
        else:
            route_weights=gumbel_softmax_st(modality_logits,self.routing_temp,self.training,self.use_gumbel,hard_route)
        if self.ablation=="no_expert_gate":
            expert_weights=torch.zeros_like(expert_logits); expert_weights[:,0]=1
        elif self.ablation.startswith("expert") and self.ablation.endswith("_only"):
            try: expert_id=int(self.ablation.replace("expert","").replace("_only",""))-1
            except Exception: expert_id=0
            expert_weights=torch.zeros_like(expert_logits); expert_weights[:,expert_id]=1
        else:
            expert_weights=gumbel_softmax_st(expert_logits,self.routing_temp,self.training,self.use_gumbel,hard_route)
        outputs=[]
        for r,(use_i,use_g,use_t) in enumerate(self.ROUTES):
            if self.ablation=="no_gene" and r in (1,3): use_g=0
            if self.ablation=="no_report" and r in (2,3): use_t=0
            # Keep image as anchor; absent auxiliary modalities are represented by zero-length token tensors.
            g=gene if use_g else gene[:,:0,:]
            t=text if use_t else text[:,:0,:]
            # The branch routing still evaluates each candidate, but only selected branches contribute in the forward pass.
            exps=[]
            for eidx in range(4):
                exps.append(self._route(r,image,g,t,eidx))
            exps=torch.stack(exps,dim=1)  # B x expert x D
            outputs.append((exps*expert_weights.unsqueeze(-1)).sum(dim=1))
        outputs=torch.stack(outputs,dim=1)  # B x route x D
        fused=(outputs*route_weights.unsqueeze(-1)).sum(dim=1)
        logits=self.classifier(fused)
        hazards=torch.sigmoid(logits).clamp(1e-6,1-1e-6)
        survival=torch.cumprod(1-hazards,dim=-1)
        risk=-survival.sum(dim=-1)
        if return_details:
            details={"modality_logits":modality_logits,"modality_weights":route_weights,"expert_logits":expert_logits,
                     "expert_weights":expert_weights,"gene_loss":gene_loss,"gene":gene_details,"risk":risk,
                     "hazards":hazards,"survival":survival,"fused":fused,"logits":logits}
            return hazards,survival,risk,gene_loss,details
        return hazards,survival,risk,gene_loss
