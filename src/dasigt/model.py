"""DASIGT model with true multi-patient batching and padding-aware attention."""
import torch
from torch import nn
import torch.nn.functional as F


def gumbel_softmax_st(logits, temperature=1.0, training=True, use_gumbel=True, hard=True):
    if training and use_gumbel:
        u = torch.rand_like(logits).clamp_(1e-8, 1.0 - 1e-8)
        g = -torch.log(-torch.log(u))
        y = F.softmax((logits + g) / max(float(temperature), 1e-5), dim=-1)
    else:
        y = F.softmax(logits / max(float(temperature), 1e-5), dim=-1)
    if not hard:
        return y
    idx = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
    return y_hard - y.detach() + y


def masked_mean(x, mask=None, dim=1):
    """Mean over valid sequence positions. mask=True means valid token."""
    if x.shape[dim] == 0:
        return x.new_zeros((x.shape[0], x.shape[-1]))
    if mask is None:
        return x.mean(dim=dim)
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CrossAttention(nn.Module):
    """Multi-head attention that ignores padded keys and zeros padded queries."""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, k=None, v=None, q_mask=None, k_mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        key_padding_mask = None if k_mask is None else ~k_mask.to(dtype=torch.bool)
        z, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.norm(q + z)
        q = self.norm2(q + self.ff(q))
        if q_mask is not None:
            q = q * q_mask.to(dtype=q.dtype).unsqueeze(-1)
        return q


class GeneModuleIdentification(nn.Module):
    """Fold-specific GMI with a pairwise eigengene-separation regularizer."""
    def __init__(self, gene_prior, train_expression, n_modules=50, temperature=0.5, power_steps=8, eps=1e-8,
                 use_prior=True, use_gene_loss=True, hidden_dim=512,
                 kmeans_n_init=10, kmeans_max_iter=300, kmeans_random_state=7):
        super().__init__()
        prior = torch.as_tensor(gene_prior, dtype=torch.float32)
        expr = torch.as_tensor(train_expression, dtype=torch.float32)
        if prior.ndim != 2 or expr.ndim != 2 or expr.shape[1] != prior.shape[0]:
            raise ValueError(f"GMI shape mismatch: prior={tuple(prior.shape)} train_expression={tuple(expr.shape)}")
        self.n_genes = prior.shape[0]
        self.n_modules = min(int(n_modules), self.n_genes)
        if self.n_modules < 2:
            raise ValueError("GMI requires at least two modules for the eigengene-separation loss.")
        self.temperature = float(temperature)
        self.power_steps = int(power_steps)
        self.eps = float(eps)
        self.use_prior = bool(use_prior)
        self.use_gene_loss = bool(use_gene_loss)
        self.register_buffer("gene_prior", prior)
        self.register_buffer("train_expression", expr)
        self.prior_shift = nn.Linear(prior.shape[1], 1, bias=False)

        # KMeans initializes the centroids; differentiable soft assignments are used in forward.
        init_device = self.prior_shift.weight.device
        init_dtype = self.prior_shift.weight.dtype
        prior_init = prior.detach().to(device=init_device, dtype=init_dtype)
        expr_init = expr.detach().to(device=init_device, dtype=init_dtype)
        with torch.no_grad():
            profiles = (expr_init + self.prior_shift(prior_init).T).T.cpu().numpy()
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.n_modules, n_init=int(kmeans_n_init),
                        max_iter=int(kmeans_max_iter), random_state=int(kmeans_random_state)).fit(profiles)
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        except Exception:
            ids = torch.linspace(0, max(0, profiles.shape[0] - 1), self.n_modules).long().numpy()
            centers = torch.tensor(profiles[ids], dtype=torch.float32)
        self.centroids = nn.Parameter(centers)
        self.gene_encoder = nn.Sequential(
            nn.Linear(self.n_modules, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, patient_expression, return_details=False):
        if patient_expression.ndim == 1:
            patient_expression = patient_expression.unsqueeze(0)
        if patient_expression.ndim != 2 or patient_expression.shape[1] != self.n_genes:
            raise ValueError(f"Expected patient_expression [B,{self.n_genes}], got {tuple(patient_expression.shape)}")

        prior = self.gene_prior
        shift = self.prior_shift(prior).squeeze(-1) if self.use_prior else torch.zeros(
            self.n_genes, device=prior.device, dtype=prior.dtype
        )
        train_x = self.train_expression + shift.unsqueeze(0)
        profiles = train_x.T  # [G, N_train]
        dist = torch.cdist(profiles.unsqueeze(0), self.centroids.unsqueeze(0)).squeeze(0).pow(2)
        assign = F.softmax(-dist / max(self.temperature, 1e-5), dim=-1)  # [G, D]
        if self.training and assign.requires_grad:
            assign.retain_grad()

        # Differentiable approximation of each module's leading singular direction.
        weighted = torch.sqrt(assign.clamp_min(self.eps))  # [G, D]
        v = torch.ones((self.n_genes, self.n_modules), device=train_x.device, dtype=train_x.dtype)
        v = v / (v.norm(dim=0, keepdim=True) + self.eps)
        pi_first_u = pi_first_v = pi_last_u = pi_last_v = None
        for step in range(self.power_steps):
            u = train_x @ (weighted * v)  # [N_train, D]
            u = u / (u.norm(dim=0, keepdim=True) + self.eps)
            v = (train_x.T @ u) * weighted  # [G, D]
            v = v / (v.norm(dim=0, keepdim=True) + self.eps)
            if step == 0:
                pi_first_u, pi_first_v = u, v
            pi_last_u, pi_last_v = u, v

        pivots = v.abs().argmax(dim=0, keepdim=True)
        pivot_values = v.gather(0, pivots)
        signs = torch.where(pivot_values < 0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values))
        v = v * signs
        loadings = weighted * v  # [G, D]

        # GE is computed for every patient in the outer-training fold, not only the current mini-batch.
        train_module_scores = train_x @ loadings  # [N_train, D], columns are eigengenes
        test_x = patient_expression + shift.unsqueeze(0)  # [B, G]
        scores = test_x @ loadings  # [B, D]
        h_gene = self.gene_encoder(scores)  # [B, H]

        if self.use_gene_loss:
            # Equation (4): 1 / sum_i sum_{j != i} ||GE_i - GE_j||_2^2.
            # torch.pdist returns each unordered pair once; multiplying by two
            # reproduces the ordered (i, j), i != j summation in the paper.
            pairwise_sq_dist = torch.pdist(train_module_scores.transpose(0, 1), p=2).pow(2)
            distance_sum = 2.0 * pairwise_sq_dist.sum()
            gene_loss = torch.reciprocal(distance_sum + self.eps)
        else:
            distance_sum = torch.zeros((), device=patient_expression.device, dtype=patient_expression.dtype)
            gene_loss = torch.zeros((), device=patient_expression.device, dtype=patient_expression.dtype)

        if return_details:
            return h_gene, gene_loss, {
                "assignment": assign,
                "module_scores": scores,
                "train_module_scores": train_module_scores,
                "loadings": loadings,
                "gene_shift": shift,
                "gene_loss_distance_sum": distance_sum,
                "pi_first_u": pi_first_u,
                "pi_first_v": pi_first_v,
                "pi_last_u": pi_last_u,
                "pi_last_v": pi_last_v,
            }
        return h_gene, gene_loss


class DASIGT(nn.Module):
    """DASIGT multimodal survival model with batch-aware variable-length token handling."""
    ROUTES = ((1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1))

    def __init__(self, gene_prior, train_expression, config, ablation=None):
        super().__init__()
        mc = config["model"]
        fc = config["features"]
        self.dim = int(mc.get("hidden_dim", 512))
        self.n_bins = int(config["survival"]["time_bins"])
        self.ablation = ablation or mc.get("ablation", "none")
        self.routing_temp = float(mc.get("routing_temperature", 1.0))
        self.use_gumbel = bool(mc.get("routing_gumbel", True))
        self.straight_through = bool(mc.get("routing_straight_through", True))
        self.use_gene_loss = self.ablation != "no_gene_loss"
        if int(mc.get("num_experts", 4)) != 4:
            raise ValueError("DASIGT architecture follows the original four-expert design; model.num_experts must be 4.")

        self.image_encoder = nn.Sequential(
            nn.Linear(fc["image_dim"], self.dim), nn.GELU(), nn.Dropout(mc.get("dropout", 0.25)), nn.Linear(self.dim, self.dim)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(fc["text_dim"], self.dim), nn.GELU(), nn.Dropout(mc.get("dropout", 0.25)), nn.Linear(self.dim, self.dim)
        )
        self.gmi = GeneModuleIdentification(
            gene_prior, train_expression, fc["gene_cluster_num"], fc["gene_temperature"],
            fc["gene_power_steps"], fc["gene_epsilon"], use_prior=self.ablation != "no_gene_prior",
            use_gene_loss=self.use_gene_loss, hidden_dim=self.dim,
            kmeans_n_init=fc.get("gene_kmeans_n_init", 10), kmeans_max_iter=fc.get("gene_kmeans_max_iter", 300),
            kmeans_random_state=fc.get("gene_kmeans_random_state", 7)
        )

        # Equation (6): modality-specific linear projection, RMSNorm, GELU, and token mean.
        self.gate_gene_proj = nn.Linear(self.dim, self.dim)
        self.gate_image_proj = nn.Linear(self.dim, self.dim)
        self.gate_text_proj = nn.Linear(self.dim, self.dim)
        self.gate_gene_norm = RMSNorm(self.dim)
        self.gate_image_norm = RMSNorm(self.dim)
        self.gate_text_norm = RMSNorm(self.dim)

        # Equation (5): two independent single-layer heads for route and expert weights.
        self.modality_gate = nn.Linear(self.dim * 3, 4)
        self.expert_gate = nn.Linear(self.dim * 3, 4)

        def attn():
            return CrossAttention(self.dim, mc.get("attention_heads", 8), mc.get("attention_dropout", 0.1))

        # EX1: CA(CA(P, R), G); EX2: CA(P, CA(R, G));
        # EX3: SA(P) + mean(R) + mean(G); EX4: FI(SA(concat[R, P, G])).
        self.experts = nn.ModuleList([
            nn.ModuleDict({"self_i": attn(), "cross1": attn(), "cross2": attn()}),
            nn.ModuleDict({"self_i": attn(), "cross1": attn(), "cross2": attn()}),
            nn.ModuleDict({"self_i": attn()}),
            nn.ModuleDict({"joint": attn()}),
        ])
        self.classifier = nn.Sequential(nn.LayerNorm(self.dim), nn.Dropout(mc.get("dropout", 0.25)), nn.Linear(self.dim, self.n_bins))

    @staticmethod
    def _ensure_mask(x, mask):
        if mask is None:
            return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return mask.to(device=x.device, dtype=torch.bool)

    def _route(self, route_idx, image, gene, text, expert_idx, image_mask, gene_mask, text_mask):
        """Apply one AMMEM expert to one image-anchored modality combination.

        For the full tri-modal route, the implementation follows Equation (8) exactly.
        For partial routes, unavailable modalities are omitted and the corresponding
        attention hierarchy is reduced to the remaining available interaction.
        """
        e = self.experts[expert_idx]
        has_gene = gene.shape[1] > 0
        has_report = text.shape[1] > 0

        if expert_idx == 0:  # EX1: CA(CA(P, R), G)
            if has_report:
                z = e["cross1"](image, text, text, q_mask=image_mask, k_mask=text_mask)
            else:
                z = image
            if has_gene:
                z = e["cross2"](z, gene, gene, q_mask=image_mask, k_mask=gene_mask)
            elif not has_report:
                z = e["self_i"](image, q_mask=image_mask, k_mask=image_mask)

        elif expert_idx == 1:  # EX2: CA(P, CA(R, G))
            if has_report and has_gene:
                report_gene = e["cross1"](
                    text, gene, gene, q_mask=text_mask, k_mask=gene_mask
                )
                z = e["cross2"](
                    image, report_gene, report_gene, q_mask=image_mask, k_mask=text_mask
                )
            elif has_report:
                # With G absent, the inner CA(R, G) reduces to the available R representation.
                z = e["cross2"](image, text, text, q_mask=image_mask, k_mask=text_mask)
            elif has_gene:
                # With R absent, use the available G representation as the inner context.
                z = e["cross2"](image, gene, gene, q_mask=image_mask, k_mask=gene_mask)
            else:
                z = e["self_i"](image, q_mask=image_mask, k_mask=image_mask)

        elif expert_idx == 2:  # EX3: SA(P) + mean(R) + mean(G)
            z = e["self_i"](image, q_mask=image_mask, k_mask=image_mask)
            if has_report:
                z = z + masked_mean(text, text_mask).unsqueeze(1)
            if has_gene:
                z = z + masked_mean(gene, gene_mask).unsqueeze(1)
            z = z * image_mask.to(dtype=z.dtype).unsqueeze(-1)

        elif expert_idx == 3:  # EX4: FI(SA(concat[R, P, G]))
            parts = []
            masks = []
            if has_report:
                parts.append(text)
                masks.append(text_mask)
            parts.append(image)
            masks.append(image_mask)
            if has_gene:
                parts.append(gene)
                masks.append(gene_mask)
            joint = torch.cat(parts, dim=1)
            joint_mask = torch.cat(masks, dim=1)
            joint_output = e["joint"](joint, q_mask=joint_mask, k_mask=joint_mask)
            image_start = text.shape[1] if has_report else 0
            z = joint_output[:, image_start:image_start + image.shape[1], :]

        else:
            raise ValueError(f"Unknown AMMEM expert index: {expert_idx}")

        # Pool the pathology-query tokens to obtain the patient-level expert output U_k.
        return masked_mean(z, image_mask)

    def forward(self, image_features, text_features, patient_expression, image_mask=None, text_mask=None, return_details=False):
        if image_features.ndim == 2:
            image_features = image_features.unsqueeze(0)
        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(0)
        if patient_expression.ndim == 1:
            patient_expression = patient_expression.unsqueeze(0)
        image_mask = self._ensure_mask(image_features, image_mask)
        text_mask = self._ensure_mask(text_features, text_mask)
        if image_features.shape[0] != text_features.shape[0] or image_features.shape[0] != patient_expression.shape[0]:
            raise ValueError("Batch-size mismatch among image, text, and gene inputs")

        image = self.image_encoder(image_features)
        text = self.text_encoder(text_features)
        # Keep padded positions zero after biased linear encoders.
        image = image * image_mask.to(image.dtype).unsqueeze(-1)
        text = text * text_mask.to(text.dtype).unsqueeze(-1)
        gene_vec, gene_loss, gene_details = self.gmi(patient_expression, return_details=True)
        gene = gene_vec.unsqueeze(1)
        gene_mask = torch.ones(gene.shape[:2], dtype=torch.bool, device=gene.device)

        # Equation (6): theta_B(E) = mean(GELU(RMSNorm(Linear(E)))).
        gene_gate = F.gelu(self.gate_gene_norm(self.gate_gene_proj(gene)))
        image_gate = F.gelu(self.gate_image_norm(self.gate_image_proj(image)))
        text_gate = F.gelu(self.gate_text_norm(self.gate_text_proj(text)))
        gene_pool = masked_mean(gene_gate, gene_mask)
        image_pool = masked_mean(image_gate, image_mask)
        text_pool = masked_mean(text_gate, text_mask)

        # Equation (5) concatenates [theta_G(GE_k), theta_P(PE_k), theta_R(RE_k)].
        gate_in = torch.cat([gene_pool, image_pool, text_pool], dim=-1)
        modality_logits = self.modality_gate(gate_in)
        expert_logits = self.expert_gate(gate_in)
        if self.training:
            if modality_logits.requires_grad:
                modality_logits.retain_grad()
            if expert_logits.requires_grad:
                expert_logits.retain_grad()

        hard_route = (not self.training) or self.straight_through
        if self.ablation in {"fixed_trifusion", "no_modality_gate"}:
            route_weights = torch.zeros_like(modality_logits)
            route_weights[:, 3] = 1
        else:
            route_weights = gumbel_softmax_st(modality_logits, self.routing_temp, self.training, self.use_gumbel, hard_route)
        if self.ablation == "no_expert_gate":
            expert_weights = torch.zeros_like(expert_logits)
            expert_weights[:, 0] = 1
        elif self.ablation.startswith("expert") and self.ablation.endswith("_only"):
            try:
                expert_id = int(self.ablation.replace("expert", "").replace("_only", "")) - 1
            except Exception:
                expert_id = 0
            expert_weights = torch.zeros_like(expert_logits)
            expert_weights[:, expert_id] = 1
        else:
            expert_weights = gumbel_softmax_st(expert_logits, self.routing_temp, self.training, self.use_gumbel, hard_route)

        outputs = []
        for r, (use_i, use_g, use_t) in enumerate(self.ROUTES):
            if self.ablation == "no_gene" and r in (1, 3):
                use_g = 0
            if self.ablation == "no_report" and r in (2, 3):
                use_t = 0
            g = gene if use_g else gene[:, :0, :]
            gm = gene_mask if use_g else gene_mask[:, :0]
            t = text if use_t else text[:, :0, :]
            tm = text_mask if use_t else text_mask[:, :0]
            exps = [self._route(r, image, g, t, eidx, image_mask, gm, tm) for eidx in range(4)]
            exps = torch.stack(exps, dim=1)  # [B, expert, H]
            outputs.append((exps * expert_weights.unsqueeze(-1)).sum(dim=1))
        outputs = torch.stack(outputs, dim=1)  # [B, route, H]
        fused = (outputs * route_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)
        hazards = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        survival = torch.cumprod(1 - hazards, dim=-1)
        risk = -survival.sum(dim=-1)

        if return_details:
            details = {
                "modality_logits": modality_logits, "modality_weights": route_weights,
                "expert_logits": expert_logits, "expert_weights": expert_weights,
                "gating_descriptor": gate_in,
                "gene_loss": gene_loss, "gene": gene_details, "risk": risk,
                "hazards": hazards, "survival": survival, "fused": fused, "logits": logits,
            }
            return hazards, survival, risk, gene_loss, details
        return hazards, survival, risk, gene_loss
