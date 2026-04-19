# r-algorithm-vs-su-algorithm
Python implementation and comparison of the **r-algorithm** and **su-algorithm** for nonsmooth and ill-conditioned optimization, with experiments on quadratic/absolute-value test functions, ε-SVR, regression error minimization, and CVaR portfolio optimization.

This project investigates how the **r-algorithm** and the **su-algorithm** solve optimization problems that are nonsmooth and potentially ill-conditioned. For such problems, classical gradient and subgradient methods often suffer from zigzagging behavior and slow convergence. The r-algorithm and the su-algorithm are designed to address these challenges more effectively and to improve convergence toward an optimal solution. The aim of this study is to compare the performance of these two algorithms through numerical experiments on several classes of problems, including designed test functions, machine learning problems, and optimization problems. Particular attention is given to the mechanism of each method: the r-algorithm accelerates the optimization process through **space dilation**, whereas the su-algorithm relies on a **subgradient-based variable metric** update. For selected experiments, the performance of the r-algorithm and su-algorithm is also benchmarked against results obtained from **Portfolio Safeguard (PSG)**.

## Project Overview

- **r-algorithm**: a subgradient-based method that improves convergence through **space dilation**
- **su-algorithm**: a variable-metric style method that updates the transformation matrix using **subgradient information**

The main goal is to understand how their different update mechanisms affect convergence speed, stability, and final solution quality across a range of numerical experiments.

## Problems Studied

The repository includes experiments for:

1. **Smooth ill-conditioned quadratic problem**

2. **Nonsmooth ill-conditioned absolute value problem**

3. **\(\varepsilon\)-Support Vector Regression (\(\varepsilon\)-SVR)**
   - Finite-sample nonsmooth regression with \(\ell_2\)-regularization

4. **Regression error minimization problem**
   - Based on minimizing the maximum of positive and negative expected residual errors

5. **CVaR portfolio optimization**
   - A constrained portfolio problem combining expected loss and Conditional Value-at-Risk under simplex constraints

## Main Findings

The numerical experiments in the report show the following general trends:

- The **r-algorithm** often converges faster and more stably than the **su-algorithm**, especially on smooth and nonsmooth ill-conditioned test problems.
- On the **quadratic** and **absolute value** problems, the r-algorithm reaches small optimality gaps in far fewer iterations.
- On **\(\varepsilon\)-SVR**, both methods perform well and achieve very similar final objective values.
- On the **regression error minimization** problem, the su-algorithm may improve faster at some early or intermediate stages, but the r-algorithm achieves a steeper final reduction.
- On the **CVaR portfolio optimization** problem, the su-algorithm is more sensitive to penalty and update parameters, while the r-algorithm behaves more robustly.
- A **memory-saving initialization** for the transformation matrix \(B_k\) reduces storage cost while preserving nearly the same optimization accuracy in the tested \(\varepsilon\)-SVR setting.

## References
1. Anton Malandii and Stan Uryasev. *Biased mean quadrangle and applications*. arXiv preprint arXiv:2603.26901, 2026.
2. Vladimir I. Norkin and Anton Y. Kozyriev. *On Shor’s r-algorithm for problems with constraints*. Cybernetics and Computer Technologies, 2023.
3. R. T. Rockafellar and S. Uryasev. *Conditional Value-at-Risk for General Loss Distributions*. Journal of Banking and Finance, 26:1443–1471, 2002.
4. Naum Z. Shor. *Nondifferentiable Optimization and Polynomial Problems*. Springer, 1998.
5. Petro Stetsyuk. *Theory and software implementations of Shor’s r-algorithms*. Cybernetics and Systems Analysis, 2017.
6. Stanislav P. Uryasev. *Adaptive variable metric algorithms for nonsmooth optimization problems*. IIASA Technical Report WP-88-60, 1988.
7. Stanislav P. Uryasev. *New variable-metric algorithms for nondifferentiable optimization problems*. Journal of Optimization Theory and Applications, 71(2):311–333, 1991.
