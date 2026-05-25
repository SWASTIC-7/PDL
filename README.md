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


 

