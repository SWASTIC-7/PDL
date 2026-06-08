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




# CASE STUDY

We have used PGLIB-OPF API-tier MATPOWER case configurations to evaluate our framework under different test senarios. We have tested our framework under datasets which are highly stressed, near-collapse grid loading profiles that traditionally precipitate mathematical divergence in conventional iterative solvers to substantiate our claims regarding reproducibility and robustness under severe system constraints.

The proposed framework is evaluated across four test sys-
tems of increasing scale and complexity: the EPRI 39-bus,IEEE
118-bus,IEEE 300-bus, and IEEE 1000-bus networks. We compare our method against three alternative initialization strategies: 1) Newton-Raphson Flat Start (NR Flat): The standard initialization heuristic (V = 1.0 p.u., θ = 0◦). 2) Supervised Neural Network Warm Start (NN Warm + NR): A GAT model trained via standard empirical Mean Squared Error (MSE) loss against pre-computed states. 3)DC Power Flow Warm Start (DCPF Warm + NR): A warm-start derived from the classical linearized DC power flow formulation


## Divergence ratio and Convergence bahaviour

The primary objective of the proposed framework is to extend the capability of stable operation under highly non-linear, heavily loaded regimes.
<figure case 39 rescue>
In the IEEE 39-bus system, a stark structural anomaly is
observed in conventional methods: NR Flat, NN Warm + NR,
and DCPF Warm + NR are rigidly bounded, converging on
exactly 770 test samples and completely failing on the remain-
ing highly stressed loading configurations. Conversely, our
proposed framework successfully resolves 1,696 test points out of 2000 highly stressed load cases which represents recovery of 75.3% that are mathematically unresolvable via standard initialization.

<figure all cases comparison on convergence count>
This advantage scales prominently as topological complexity increases. For the IEEE 118-bus network, $\mathtt{PDL\text{-}Warm+NR}$ achieves a perfect convergence profile (1,600 out of 1,600 samples), while the baselines diverge on 257 distinct configurations. In the highly complex IEEE 300-bus system, severe non-linear stress causes the standard $\mathtt{NR\ Flat}$ and $\mathtt{DCPF\ Warm+NR}$ to crash, successfully resolving only 69 points. The standard $\mathtt{NN\ Warm+NR}$ degrades even further to 39 points due to unconstrained out-of-bounds initialization predictions. Meanwhile, the physics-biased $\mathtt{PDL\text{-}Warm+NR}$ maintains a dominant convergence footprint, securing 618 stable solutions under identical near-collapse loads.


## Average iteration Reduction

To quantify algorithmic acceleration, Fig.~\ref{fig:avg_iterations} records the average number of Newton-Raphson iterations required to reach a tight convergence tolerance ($\epsilon = 10^{-4}$), computed exclusively over the subset of test samples where all methods successfully converged.

<figure>

For the baseline converged subset, the proposed $\mathtt{PDL\text{-}Warm+NR}$ provides a trajectory significantly closer to the true solution manifold. In the IEEE 39-bus case, our framework exhibits a 26.5\% reduction in iteration counts relative to a flat start ($3.94$ iterations versus $5.36$), systematically outperforming the unconstrained MSE model ($4.49$ iterations).

Notably, in the larger 118-bus and 300-bus configurations, the $\mathtt{DCOPF\ Warm+NR}$ baseline shows marginally fewer iterations ($4.15$ and $4.33$, respectively) than the proposed method ($4.25$ and $4.58$, respectively) on easily converged samples. However, this marginal performance advantage is an artifact of DCPF's strict linear assumptions, which discard reactive power ($Q$) and voltage magnitude ($V$) variations. While this simplification yields slight gains for nominal, low-stress loading profiles, it lacks the modeling capacity to handle non-linear voltage collapse boundaries. Consequently, this superficial iteration efficiency vanishes globally, as DCPF completely fails to rescue the large volume of highly stressed grid configurations that our non-linear $\mathtt{PDL}$ framework successfully maps.


## Inference and convergence time analysis

A common drawback of neural-network-accelerated solvers is the computational latency introduced during the forward inference pass. To address this, we define the total execution time as the sum of neural network inference latency and downstream solver convergence time.

<figure>

Because our physics-biased graph attention layer is implemented via highly optimized tensor-parallel operations, the inference overhead is negligible ($\approx 0.04\,\text{s}$ for Case 39 and $\approx 0.12\,\text{s}$ for Case 118). By drastically minimizing the downstream iteration budget, the proposed pipeline reduces total execution time by up to 26.7\% ($80.45\,\text{s}$ for $\mathtt{PDL\text{-}Warm+NR}$ versus $109.80\,\text{s}$ for $\mathtt{NR\ Flat}$ in the IEEE 39 system). This demonstrates that the framework achieves computational acceleration without sacrificing physical fidelity.

## Robustness under Multivariant Data stress

To rigorously map the generalization limits of the structural self-attention layers, the network is subjected to four distinct multivariant stress strategies designed to mimic extreme, real-world grid volatility:
\begin{enumerate}
    \item \textit{Stochastic Load Noise}: Random active and reactive load perturbations spanning standard high-variance bounds $[0.75, 1.15]$.
    \item \textit{Systemic Load Upscaling}: Uniform active and reactive power scaling from $[1.0, 1.3]$ to simulate extreme peak demand.
    \item \textit{Systemic Load Downscaling}: Depressed loading fields downscaled to $[0.75, 1.0]$.
    \item \textit{Reactive Over-Excitation ($Q$-Stress)}: Selective, high-stress reactive power inflation spanning $[1.2, 2.0]$ to simulate severe inductive grid choking.
\end{enumerate}

<figure>

As shown in Fig.~\ref{fig:stress_strategy}, the proposed physics-biased architecture maintains stable convergence across these distribution shifts. It is worth noting that for the IEEE 300-bus system, nominal profiles converge relatively easily across baseline models. Thus, our discussion focuses heavily on the highly non-linear, anomalous response regions of Case 39 and Case 118, where unconstrained neural network representations fail to generalize.


## Stress sweep rescue capabilities

Fig.~\ref{fig:stress_sweep} illustrates a continuous parametric stress sweep evaluated at discrete scaling operational points: $0.6, 0.7, 0.8, 0.9, 1.0, 1.1,$ and $1.2$. 

<figure>

This continuous sweep traces the exact breakdown threshold where the geometric contraction mapping of the Newton-Raphson method fails. The tracking curves demonstrate that while standard MSE networks degrade rapidly as the load factor moves away from nominal training distributions, our self-supervised primal-dual formulation tracks the underlying solution manifold smoothly, preserving numerical stability well into severe over-loading conditions.