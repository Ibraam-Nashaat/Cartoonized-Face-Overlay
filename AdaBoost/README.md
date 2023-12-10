# Example algorithm (Discrete AdaBoost)
With:
* Samples $x_1 \dots x_n$
* Desired outputs $y_1 \dots y_n, y \in \{-1, 1\}$
* Initial weights $w_{1,1} \dots w_{n,1}$ set to $\frac{1}{n}$
* Error function $E(f(x), y, i) = e^{-y_i f(x_i)}$
* Weak learners $h\colon x \rightarrow \{-1, 1\}$
For $t$ in $1 \dots T$:
* Choose $h_t(x)$:
** Find weak learner $h_t(x)$ that minimizes $\epsilon_t$, the weighted sum error for misclassified points $\epsilon_t = \sum^n_{\stackrel{i=1}{h_t(x_i)\neq y_i}} w_{i,t} $
** Choose $\alpha_t = \frac{1}{2} \ln \left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
* Add to ensemble:
** $F_t(x) = F_{t-1}(x) + \alpha_t h_t(x)$
* Update weights:
** $w_{i,t+1} = w_{i,t} e^{-y_i \alpha_t h_t(x_i)}$ for $i$ in $1 \dots n$
** Renormalize $w_{i,t+1}$ such that $\sum_i w_{i,t+1} = 1$
