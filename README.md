The observations to be written in the report

# Warm-starting Newton-Raphson for Power Flow using Self-Supervised Primal-Dual learning

## Key point to highlight in the report

- Explain the self-supervised learning and how are we achieving this in Power flow
- Doing warm start, hence accuracy of the answer is no question here
- Uses PGLIB-OPF API-tier MATPOWER cases (near-collapse / stressed) downloaded + cached automatically (great for paper reproducibility claims).
- GAT architecture here: it learns bus-to-bus interactions directly over the physical grid graph (adjacency + edge weights from |Y_{bus}|), so the model's message passing follows electrical connectivity and can adaptively focus attention on the most influential neighboring buses—making the warm-start robust under stressed operating conditions.
- all the metrics to highlight

We are using case 39, 118, 300, 1000 (still needed to be run)

- Conversion ratio is higher in our case
- Less no of average iteration required in our case
- Less convergence time in our case
- Better stress test result than normal NR

## Divergence ratio

![](./assets/chart_convergence_count_300.png)
![](./assets/chart_convergence_count_118.png)
![](./assets/chart_convergence_count_39.png)
These chart contains the convergence count for each cases
We have total 4 cases here

- Newton Raphson flat start
- Newton Raphson warm start with PDL NN (ours)
- Newton Raphson warm start with NN with MSE loss
- Newton Raphson warm start with dcpf
  Here we will emphasize more on case 39 and 300, specifically pointing out to case 39 (as this case is not normal)

## Average iteration

![](./assets/chart_method_iterations_300.png)
![](./assets/chart_method_iterations_118.png)
![](./assets/chart_method_iterations_39.png)

This chart shows the average iteration count only for the converged test data
Here dcpf convergance iteration count is slightly lesser for case 300 and 118, need to figure out how to present this data -- (**ask Parikshit sir**)

## Average time

![](./assets/chart_method_times_300.png)
![](./assets/chart_method_times_118.png)
![](./assets/chart_method_times_39.png)

This average iteration is the time taken to converge the test cases
For flat start it is just the newton raphson time
For Warm start it is the sum of inference time + the convergence time

## Rescue bar

![](./assets/chart_rescue_bar_300.png)
![](./assets/chart_rescue_bar_118.png)
![](./assets/chart_rescue_bar_39.png)

These rescue bar highlights specifically the test points rescued with respect to normal flat NR, can specifically use for case 39 only

## Different test data stress

![](./assets/chart_strategy_300.png)
![](./assets/chart_strategy_118.png)
![](./assets/chart_strategy_39.png)

This data shows how our pdl also converges the test data points when they are diverged in 4 ways

- first is the +- noise in between [0.75, 1.15]
- Second is the upscaling  [1, 1.3]
- third is the downscaling [0.75, 1]
- fourth is the stressing only the Q value [1.2, 2]

here case 300 converges in flat NR also for most of the cases, therefore need not to be shown
(**needed to change the labels of the graph appropriately and then will explain each label at the introduction of image**)

## Stress sweep

![](./assets/chart_stress_sweep_300.png)
![](./assets/chart_stress_sweep_118.png)
![](./assets/chart_stress_sweep_39.png)

These graphs shows the sweep test with sweep points 0.6, 0.7. 0.8, 0.9, 1, 1.1, 1.2
(**Needed to discuss with parikshit sir which cases to be shown in report**)

## participation factor

![](./assets/chart_participation_dots_300.png)
![](./assets/chart_participation_dots_118.png)
![](./assets/chart_participation_dots_39.png)

(**needed to discuss with parikshit sir -- what to do whith this**)

We also have the graph which can show the data across all the cases (bus-system), for average iteration count and inference time -- (**need to fix those graph, but it onlu compare with NR**)

---

# Method (PDL-GAT) — Mathematics and Implementation Details

This section documents the exact model and training objective used in `teb.py`.

## 1) Problem Setup (AC Power Flow Constraints)

For each operating point (scenario), the inputs are per-bus demands:

$$
P_d \in \mathbb{R}^{N},\quad Q_d \in \mathbb{R}^{N}
$$

The primal network predicts a feasible warm-start state (in per-unit):

$$
x = (P_g, Q_g, V, \theta)
$$

where:

- $P_g, Q_g$ are generator outputs (only defined on generator buses, but assembled into bus injections in code),
- $V \in \mathbb{R}^{N}$ is voltage magnitude,
- $\theta \in \mathbb{R}^{N}$ is voltage angle (slack angle fixed to 0 in the architecture).

The complex bus voltages are

$$
V_c = V \odot (\cos\theta + j\sin\theta) = V \odot e^{j\theta}
$$

With bus admittance matrix $Y_{bus} \in \mathbb{C}^{N\times N}$, the injected complex power is computed (as implemented) as

$$
S = V_c \odot \overline{(Y_{bus} V_c)}
$$

so

$$
P(x) = \Re(S),\quad Q(x) = \Im(S)
$$

Net injections are formed in code as

$$
P_{inj} = P_g - P_d,\quad Q_{inj} = Q_g - Q_d
$$

The (per-bus) power-balance residuals are

$$
r_P(x) = P(x) - P_{inj},\quad r_Q(x) = Q(x) - Q_{inj}
$$

Slack-bus residuals are set to 0 in `compute_power_balance(...)` (slack absorbs mismatch).

We stack residuals as

$$
r(x) = \begin{bmatrix} r_P(x) \\ r_Q(x) \end{bmatrix} \in \mathbb{R}^{2N}
$$

## 2) Graph Construction (What the GAT “sees”)

We build a graph over buses from the physical network:

- Adjacency mask $A \in \{0,1\}^{N\times N}$ from nonzero $Y_{bus}$ (diagonal forced to 1 so each bus can attend to itself).
- Edge-weight matrix $W \in \mathbb{R}^{N\times N}$ from $|Y_{bus}|$ (normalized by its max).

These two objects are passed into the attention layers:

- $A$ is a *hard mask* (disconnected nodes get attention logit $-\infty$),
- $W$ is an *additive bias* in the attention logits (scaled by learned per-head scalars).

## 3) Graph Attention (GAT) Layer — Equations Matching `teb.py`

The implementation is masked multi-head scaled dot-product attention on the grid graph.

Let $x \in \mathbb{R}^{B\times N\times C}$ be node embeddings (batch $B$, buses $N$, model dim $C$). With $H$ heads and head-dim $D=C/H$:

1) Linear projections (single projection producing Q,K,V):

$$
[Q,K,V] = x W_{qkv},\quad W_{qkv} \in \mathbb{R}^{C\times 3C}
$$

reshaped to

$$
Q,K,V \in \mathbb{R}^{B\times H\times N\times D}
$$

2) Attention logits per head:

$$
S^{(h)} = \frac{Q^{(h)} (K^{(h)})^\top}{\sqrt{D}}\in\mathbb{R}^{B\times N\times N}
$$

3) Graph mask:

$$
S^{(h)}_{ij} \leftarrow -\infty\ \text{if}\ A_{ij}=0
$$

4) Edge-weight bias (learned per-head scalar $\gamma_h$):

$$
S^{(h)}_{ij} \leftarrow S^{(h)}_{ij} + \gamma_h\,W_{ij}
$$

5) Softmax over neighbor dimension:

$$
\alpha^{(h)}_{ij} = \mathrm{softmax}_j\left(S^{(h)}_{ij}\right)
$$

Dropout is applied on $\alpha$.

6) Message aggregation:

$$
Z^{(h)}_i = \sum_j \alpha^{(h)}_{ij} V^{(h)}_j
$$

Concatenate heads and apply output projection:

$$
\mathrm{Attn}(x) = W_o\,[Z^{(1)};\dots;Z^{(H)}]
$$

### GATBlock

Each `GATBlock` is a residual pre-norm block:

$$
x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))
$$

$$
x \leftarrow x + \mathrm{FFN}(\mathrm{LN}(x))
$$

## 4) Primal and Dual Networks (What They Output)

### Primal network: `ACPFPrimalGAT`

Inputs: $(P_d, Q_d)$ plus static grid information (generator limits, PV/slack voltage setpoints, $A$, $W$).

Outputs: $(P_g, Q_g, V, \theta)$ with structural constraints enforced:

- Generator bounds via sigmoid scaling:

$$
P_g = P_{min} + \sigma(\cdot)\,(P_{max}-P_{min}),\quad Q_g = Q_{min} + \sigma(\cdot)\,(Q_{max}-Q_{min})
$$

- Voltage magnitudes:
  - PV + slack buses are clamped to setpoints,
  - PQ buses are predicted in a fixed range:

$$
V_{pq} = 0.85 + 0.30\,\sigma(\cdot)\ \in [0.85,1.15]
$$

- Angles:
  - $\theta_{slack}=0$,
  - non-slack predicted with a tanh cap:

$$
\theta_i = \tanh(\cdot)\,\theta_{max}\ \in [-\theta_{max},\theta_{max}]
$$

After several graph-attention blocks, the model also applies a *global* (dense) multi-head attention across all buses to capture long-range coupling.

### Dual network: `ACPFDualGAT`

Inputs: $(P_d,Q_d)$ and the same graph.

Output: Lagrange multipliers for power balance constraints:

$$
\lambda = \begin{bmatrix}\lambda_P\\\lambda_Q\end{bmatrix} \in \mathbb{R}^{2N}
$$

In code, the dual head predicts 2 values per bus and then concatenates them into a length-$2N$ vector.

## 5) PDL Objective (Augmented Lagrangian)

The primal network is trained to reduce constraint violation using an augmented Lagrangian:

$$
\mathcal{L}_\rho(x,\lambda) = \lambda^\top r(x) + \frac{\rho}{2}\,\lVert r(x)\rVert_2^2
$$

with penalty weight $\rho>0$.

This is exactly implemented as

- a linear term $\lambda_P^\top r_P + \lambda_Q^\top r_Q$,
- plus the quadratic penalty $(\rho/2)(\lVert r_P\rVert_2^2 + \lVert r_Q\rVert_2^2)$.

### Why “self-supervised” here?

No Newton–Raphson labels are required to train the PDL model:

- The residual $r(x)$ is computed directly from physics ($Y_{bus}$) and predicted state $x$.
- The dual network targets are generated from a dual-ascent style update (next section).

## 6) Dual Update Target (Learned Dual Ascent)

The classical augmented-Lagrangian dual ascent step is

$$
\lambda^{k+1} = \lambda^{k} + \rho\,r(x^{k+1})
$$

Instead of storing a separate $\lambda$ for every training sample, we learn a function $\lambda_\phi(P_d,Q_d)$.

In `train_epoch(...)`, we create a frozen copy of the previous dual net to compute

$$
\lambda^{k} \approx \lambda_{\phi_{old}}(P_d,Q_d)
$$

and then build the supervised target

$$
\lambda_{target} = \lambda_{\phi_{old}}(P_d,Q_d) + \rho\,r(x)
$$

The dual net is trained with MSE to match this target.

## 7) Deterministic / Adaptive \(\rho\) Schedule (as coded)

Let $v_k$ be the maximum absolute constraint violation observed on a held-out subset in outer iteration $k$:

$$
v_k = \max\big(\lVert r_P(x)\rVert_\infty,\ \lVert r_Q(x)\rVert_\infty\big)
$$

After a warmup, every `rho_check_freq` steps:

If

$$
v_k > \tau\,v_{k-1}
$$

then

$$
\rho \leftarrow \min(\alpha\rho,\rho_{max})
$$

This increases the penalty only when violation is not decreasing fast enough.

## 8) Pseudocode (PDL Algorithm as Implemented)

### 8.1 Graph attention layer

```text
function GRAPH_ATTENTION_LAYER(x, A, W):
	# x: (B, N, C), A: (N, N) adjacency mask, W: (N, N) edge weights (optional)
	QKV = Linear_qkv(x)                          # (B, N, 3C)
	Q, K, V = reshape_to_heads(QKV)              # each: (B, H, N, D)

	S = (Q @ K^T) / sqrt(D)                      # (B, H, N, N)
	S[A == 0] = -inf                             # hard graph mask

	if W is not None:
		S = S + W * edge_scale_per_head          # additive bias

	alpha = softmax(S, dim = neighbor_j)
	alpha = nan_to_num(alpha, 0)
	alpha = dropout(alpha)

	Z = alpha @ V                                # (B, H, N, D)
	Z = concat_heads(Z)                          # (B, N, C)
	return out_proj(Z)
```

### 8.2 One outer training epoch: `PDL_ACPF_GAT.train_epoch`

```text
function PDL_TRAIN_EPOCH(P_batch, Q_batch, inner_iters, batch_size, accum_steps):

	eff = max(1, floor(inner_iters / accum_steps))

	# ----- Primal step (minimize augmented Lagrangian) -----
	repeat eff times:
		shuffle data indices
		zero_grad(primal_optimizer)
		for minibatch (Pm, Qm):
			(Pg, Qg, V, theta) = primal_net(Pm, Qm, A, W)
			lambda = stop_gradient( dual_net(Pm, Qm, A, W) )

			(rP, rQ) = power_balance_residual(Pg, Qg, V, theta, Pm, Qm)

			L_primal = mean( lambdaP · rP + lambdaQ · rQ
							 + (rho/2) * (||rP||_2^2 + ||rQ||_2^2) )

			backprop(L_primal / accum_steps)
			if accumulation boundary reached:
				clip_grad(primal_net)
				primal_optimizer.step(); zero_grad(primal_optimizer)
		primal_scheduler.step()

	# Cache old dual net outputs for all samples
	dual_old = frozen_copy(dual_net)
	lambda_old_all = dual_old(P_batch, Q_batch, A, W)   # computed in chunks

	# ----- Dual step (fit dual-ascent target) -----
	repeat eff times:
		shuffle data indices
		zero_grad(dual_optimizer)
		for minibatch indices bi:
			(Pm, Qm) = (P_batch[bi], Q_batch[bi])
			(Pg, Qg, V, theta) = stop_gradient( primal_net(Pm, Qm, A, W) )

			lambda = dual_net(Pm, Qm, A, W)
			lambda_old = lambda_old_all[bi]

			(rP, rQ) = power_balance_residual(Pg, Qg, V, theta, Pm, Qm)
			lambda_target = lambda_old + rho * concat(rP, rQ)

			L_dual = MSE(lambda, lambda_target)
			backprop(L_dual / accum_steps)
			if accumulation boundary reached:
				clip_grad(dual_net)
				dual_optimizer.step(); zero_grad(dual_optimizer)
		dual_scheduler.step()

	# ----- Measure violation + update rho -----
	max_viol = max_abs_violation_on_subset()
	maybe_update_rho(max_viol)  # warmup + periodic check

	return (mean_primal_loss, max_viol)
```
