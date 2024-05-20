# Sleep

### Biological Functions of Sleep

NREM Sleep:

1. Transfer memory trace from hippocampus to cortex [@klinzing2019mechanisms];
2. Memory consolidation, more for declarative memory;
3. Priming, promotes the capture and translation of plasticity-related products to tagged synapses [@seibt2019primed].

REM Sleep:

1. Memory consolidation, more for procedural, spatial, and emotional memory;
2. Forgetting [[ref]](#Crick1983);
3. Captured plasticity-related products are translated into proteins to promote the final stage of structural plasticity stabilization [[ref]](#Seibt2019).

### Algorithm Framework

The goal is to find a causal directed acyclic diagram (DAG) $G$ that describes the environment

$$G = \{V,U,F,P(u)\} \; ,$$

where its observed variables $V$ represent joints, sensors, etc.; $U$ represent unobserved environment components, learned concepts, etc; and its edges $F:V\leftarrow f_V(Pa_V,U_V)$ are structural functions that assign values to each variable based on its parents.

On day k, Wake: If available, execute the optimal sequence of actions, and orient possible arrows in the PAG based on the result. And then execute some random actions

$$A=\{A_i\}_{i=1}^{N_{\text{seq}}}, \; A_i=\{A_{i,t}\}_{t=1}^{T_i} \; ,$$

where the space of the $j$-th joint is $A_{i,t,j} \in \mathcal{A} = \{-1,0,1\}$, and -1,0,1 mean backward, hold, forward. The sensor sequences after each action step are

$$X=\{X_i\}_{i=1}^{N_{\text{seq}}}, \; X_i=\{X_{i,t}\}_{t=1}^{T_i} \; ,$$

where the space of the $j$-th sensor observation is $X_{i,t,j} \in \mathcal{X}^{(j)} = \mathbb{R}^{D_j}$.

On night k, NREM cycle 1: During sleep, the robot does not move, and is doing some "mental" calculation. <!--Its energy level decays slowly and linearly with time.--> $G_k$ is obtained by running a causal discovery algorithm that allows hidden nodes and selection nodes (such as FCI [[ref]](#PC2000), RFCI [[ref]](#Colombo2012), or GFCI [[ref]](#Ogarrio2016)), where if available, each action is weighted by the difference between expected results vs. observed result. The result is represented as a Partial Ancestral Graph (PAG).
FCI-JCI?

On night k, REM: The optimal sequence of actions is derived based on PAG together with the expected results under the actions. The sequence of actions and expected results are described as a tree.

On night k, NREM cycle 2: Some hidden nodes (representing concepts) were added to the graph and fitted. Likelihood ratio test is used to decide whether to keep the additional hidden nodes.

### References

1. <a name="Crick1983" href=http://www.rctn.org/vs265/crick-mitchison-sleep.pdf>Crick, F. and Mitchison, G., 1983. The function of dream sleep. Nature, 304(5922), pp.111-114.</a>
1. <a name="PC2005" href=https://doi.org/10.1007/978-1-4612-2748-9>Spirtes, P., Glymour, C.N., Scheines, R. and Heckerman, D., 2000. Causation, prediction, and search. MIT press.</a> PC is based on the fact that under the causal Markov condition and the faithfulness assumption, when there is no latent confounder, two variables are directly causally related (with an edge in between) if and only if there does not exist any subset of the remaining variables conditioning on which they are independent.
1. <a name="Glymour2019" href=https://doi.org/10.3389/fgene.2019.00524>Glymour, C., Zhang, K. and Spirtes, P., 2019. Review of causal discovery methods based on graphical models. Frontiers in genetics, 10, p.524.</a>
1. <a name="Colombo2012" href=https://www.jstor.org/stable/41713636>Colombo, D., Maathuis, M.H., Kalisch, M. and Richardson, T.S., 2012. Learning high-dimensional directed acyclic graphs with latent and selection variables. The Annals of Statistics, pp.294-321.</a>
1. <a name="Colombo2012" href=https://www.jstor.org/stable/41713636>Colombo, D., Maathuis, M.H., Kalisch, M. and Richardson, T.S., 2012. Learning high-dimensional directed acyclic graphs with latent and selection variables. The Annals of Statistics, pp.294-321.</a>
1. <a name="Ogarrio2016" href=http://proceedings.mlr.press/v52/ogarrio16.html>Ogarrio, J.M., Spirtes, P. and Ramsey, J., 2016, August. A hybrid causal search algorithm for latent variable models. In Conference on probabilistic graphical models (pp. 368-379). PMLR.</a>


\bibliography

