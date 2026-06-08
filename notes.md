So we have few things to highlight

## Self supervised learning

(here i explain why and hwo self supervised learning, explains methamatically how are we acheiving this)

Conventional deep-learning-accelerated power flow methods depend heavily on pre-computed Newton-Raphson solutions. Generating these solutions introduces offline computational bottlenecks and fundamentally fails under heavily loaded grid configurations (where the underlying solver diverges), leaving the model reliant on the classical solver's capabilities.

To remove this data dependency, we propose a framework that is purely physics-driven and self-supervised using a Primal-Dual methodology. Instead of minimizing errors against empirical target values, our framework uses the network admittance matrix ($Y_{\text{bus}}$) to provide continuous guidance toward the true solution. To achieve this, we formulate the power flow problem as an augmented Lagrangian optimization process.

### Augmented Langragian Primal Dual in Power flow
The primal network maps the per-bus active and reactive power demands (Pd, Qd) to an estimated operational state vector x = (Pg , Qg , V, θ). 
A differentiable physics layer native to the
architecture assembles complex bus voltages via Vc = V ⊙ejθ
and derives node active and reactive injections via a tensor
implementation of Ohm’s and Kirchhoff’s laws:
<equation>
The instantaneous physical power balance mismatch residuals
r(x) = [rP (x)⊤, rQ(x)⊤]⊤, where rP (x) = P (x)−(Pg −Pd)
and rQ(x) = Q(x) − (Qg − Qd), are evaluated dynamically. Then these are passed to the primal networks' loss function: 
<equation>
whereas the dual network is trained via Mean Squared Error (MSE) to track and predict the approximate Lagrange multipliers λ across changing load distributions.
<equation of dual loss>
Then we update the (lambda) by using the historical dual network weights ϕold : 
<equation>


## Convergence stability and Guaranteed accuracy

(why can we diverge, how we fix it)

Since our primal-dual framework trains an interactive network structure from scratch without needing offline target values, can suffer from intial optimization instability. The primal network and dual network are coupled tightly, a wild guess from primal will motivate dual to calculate massive value for its multiplier (lambda), which eventually will make the gradients explode, causing the model to diverge, even before understanding the network structure.

### Decoupled primal Pre-training

To ensure structural stability, we isolate the primal network during pre-trianing period by making (lambda =0 ). Now the primal loss is governed by an initial penalty coefficient (rho pre). The optimization objective simplifies to minimizing the mean squared residual violations across all buses:
<equation>

During this period, primal network learns basic electrical consistency,  such as the spatial propagation of voltage drops and phase angle distributions 

### Dynamic rho-tracking and guaranteed accuracy

Once this pre train is completed (we take 15 iterations of pre-train) the dual network becomes active, optimization shifts to cmplete augmented langrangian formulation. To guarantee that the
framework converges strictly within the tight tolerances required by industrial power system operations (e.g., ϵ ≤ 10−4 p.u.), we implement a deterministic monitoring schedule that tracks the maximum boundary violation across the system:
<equation>
At fixed epoch intervals, we compare the current maximum violation against previous moving average scaled by decay factor τ ∈ (0, 1). If the 
constraint satisfaction rate is violated (V(t)
max > τ · V(t−1)
max ), then the
penalty multiplier is updated via an aggressive expansion step:
<equation>
where α > 1 represents the penalty acceleration factor.
Dynamically scaling ρ functions as a structural guarantee. It
compresses the remaining residual boundaries toward zero,
ensuring that the self-supervised outputs provide the exact
same level of physical accuracy as a fully converged classical
Newton-Raphson solver.





## Physics Biased Graph Attention Network (GAT) Architechture

Alternating current (AC) power grids are highly complex network systems. Under extreme stress, changes in one part of the grid can instantly cause non-linear disruptions far away in the grid. Standard networks struggle because power grids don't have neat, uniform topology. Also basic Graph Attention Networks (GATs) struggle because they tend to mix node data too aggressively (over-smoothing) or miscalculate how power actually flows. To fix this, we developed a Physics Biased Graph Attention Network (ACPFPrimalGAT). By integrating the laws of electrical engineering directly into the network's attention layers, our model successfully tracks these complex, widespread grid dependencies without losing physical constraints.


### Structural Inductive Bias and Adjacency Masking: 
To enforce the physical constraints, we force the network's message passing layers to follow the actual topology of the power grid by introducing a Structural Inductive Bias ie, assumptions based on physical shape, topology, or connectivity. We model the electrical grid as a undirected graph G=(V,E), where the nodes V corresponds to the set of buses N=|V| and the edges E corresponds the physical transmission lines. Instead of letting the self-attention mechanism calculate random correlations between unrelated or distant buses, we limit the model's focus. The attention boundaries are strictly dictated by the zero and non-zero patterns of the complex bus admittance matrix Ybus ∈ CN ×N, meaning the model can only pass information along real, physical connections.

We construct the base topology of our network using a spatial graph adjacency mask, A ∈ {0, 1}N ×N which maps out local connections and includes self-loops for each bus. Mathematically, it filters communication paths by checking the values within the admittance matrix:
<equation>

### Physical Scale Injection and Attention Formulation:
Masking the connections, however, isn't enough when the grid faces near collapse loads or high voltage stress, as the model can lose awarerness of physical network scales. We fix this by modifying the raw dot-product attention weights directly. We scale these values using an absolute admittance magnitude matrix W ∈ RN ×N, normalized against the highest admittance value found in the system as:
<equation>



Within each attention layer l, the model updates its understanding of the grid by calculating attention weights between nodes. For a hidden node embedding tensor x ∈ RB×N ×C  at layer l, the multi-head scaled dot-product attention weights $S^{(h)}_{ij}$ for each attention head $h$ are strictly limited to adjacent buses using our adjaceny mask $\mathbf{A}$:
<equation>
where W(h)
q , W(h)
k ∈ RC×D denote the trainable linear projection weights for queries and keys, respectively, and $\gamma_h$ is a head-specific trainable scalar variable that dynamically scales the influence of the physical transmission line weights from our normalized admittance matrix.

The directional edge weights α(h)ij  are subsequently normalized within spatial neighbor domains using a masked softmax operation:
<equation>
Once these weights are settled, each node aggregates the hidden features from its connected neighbors. The updated features are then accumulated across all $H$ attention heads, ensuring that localized message propagation respects basic structural grid properties:
<equation>
<equation>
By binding the graph attention layers directly to the physical grid characteristics via $\mathbf{A}$ and $\mathbf{W}$, the estimated state variables $x = [P_g, Q_g, V, \theta]^\top$ generated by the network remain structurally aligned with the underlying grid physics. This spatial biasing dramatically reduces structural anomalies, allowing the self-supervised framework to maintain stable convergence even under highly stressed, edge-of-collapse load profiles.









