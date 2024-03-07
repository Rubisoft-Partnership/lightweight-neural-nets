# lightweight-neural-nets

Neural network models for embedded devices


## Model-ff

Model-ff uses Forward-Forward algorithm to train a neural network.

### Definitions

$\theta \in \R: \text{threshold}$

$\alpha \in \R: \text{learning rate}$

$f \in\N: \text{features}, l\in\N: \text{classes}$

$\mathcal{L} = \{l_1, ..., l_l\}: \text{set of labels}$

$u_0\in\N: \text{hidden layer size}, u_1\in\N: \text{output layer size}$

$x\in\R^f: \text{sample}$

$y\in\R^l: \text{label}$

$\eta:\R^f\times\R^l\rightarrow\R^f: \text{embed function}$

$W^{\tiny{(0)}}\in\R^{f\times u_0}: \text{weight matrix from input to hidden layer}$

$W^{\tiny{(1)}}\in\R^{u_0\times u_1}: \text{weight matrix from hidden to output layer}$

$a^{\tiny{(0)}}:\R^{u_0}\rightarrow\R^{u_0}: \text{hidden activation function (ReLu)}$

$a^{\tiny{(1)}}:\R^{u_1}\rightarrow\R^{u_1}: \text{output activation function (ReLu)}$

$G(z):\R^{u_1}\rightarrow\R=\sum_{i=1}^{i\le u_1}(z^{\tiny{(1)}}_{i})^2$

$L(z, \bar{z}):\R^{u_1 \times u_1}\rightarrow\R=\zeta\bigl(\theta - G(z^{\tiny{(1)}})\bigr) + \zeta\bigl(G(\bar{z}^{\tiny{(1)}})-\theta)$


### Forward pass

Input to hidden layer:

$x :=\eta(x',y)$

$\bar x :=\eta(x,\bar{y}), \bar{y}\in\mathcal{L}|\bar{y}\neq y$

$h^{\tiny{(0)}} := W^{\tiny{(0)}}x'$

$\bar{h}^{\tiny{(0)}} := W^{\tiny{(0)}}\bar x$

$z^{\tiny{(0)}} := a^{\tiny{(0)}}(h^{\tiny{(0)}})$

$\bar{z}^{\tiny{(0)}} := a^{\tiny{(0)}}(\bar{h}^{\tiny{(0)}})$

Hidden to output layer:

$h^{\tiny{(1)}} := W^{\tiny{(1)}}z^{\tiny{(0)}}$

$\bar{h}^{\tiny{(1)}} := W^{\tiny{(1)}}\bar{z}^{\tiny{(0)}}$

$z^{\tiny{(1)}} := a^{\tiny{(1)}}(h^{\tiny{(1)}})$

$\bar{z}^{\tiny{(1)}} := a^{\tiny{(1)}}(\bar{h}^{\tiny{(1)}})$

### Backward pass

Updating hidden layers weights:

$
w^{\tiny{(1)}}_{i, j} =
w^{\tiny{(1)}}_{i, j} -\alpha\frac{\delta L}{\delta w^{\tiny{(1)}}_{i, j}} =
w^{\tiny{(1)}}_{i, j} -
\alpha\Bigl(\frac{\delta L_{pos}}{\delta w^{\tiny{(1)}}_{i, j}} + 
\frac{\delta L_{neg}}{\delta w^{\tiny{(1)}}_{i, j}}\Bigr)
$

$
\frac{\delta L_{pos}}{\delta w^{\tiny{(1)}}_{i, j}} = 
\frac{\delta L_{pos}}{\delta G(z^{\tiny{(1)}})}
\frac{\delta G(z^{\tiny{(1)}})}{\delta z^{\tiny{(1)}}_{j}} 
\frac{\delta z^{\tiny{(1)}}_{i}}{\delta w^{\tiny{(1)}}_{i,j}}
$

$
\frac{\delta L_{pos}}{\delta G(z^{\tiny{(1)}})} =
-\sigma\bigl(\theta - G(z^{\tiny{(1)}})\bigr)
$

$
\frac{\delta G(z^{\tiny{(1)}})}{\delta z^{\tiny{(1)}}_{j}} =
2z^{\tiny{(1)}}_{j}
$

Using ReLu activation function:

$
\frac{\delta z^{\tiny{(1)}}_{j}}{\delta w^{\tiny{(1)}}_{i,j}} =
\frac{\delta a^{\tiny{(1)}}\bigl(\sum_iw^{\tiny{(1)}}_{i,j} z^{\tiny{(0)}}_i\bigr)}{\delta w^{\tiny{(1)}}_{i,j}} =
\begin{cases}
    z^{\tiny{(0)}}_i & \text{if} \space h^{\tiny{(1)}}_j \ge 0 \\
    0 & \text{if} \space h^{\tiny{(1)}}_j \lt 0 \\
\end{cases}
$

$
\frac{\delta L_{pos}}{\delta w^{\tiny{(1)}}_{i, j}} =
\begin{cases}
    -\sigma\bigl(\theta - G(z^{\tiny{(1)}})\bigr)2z^{\tiny{(1)}}_{j}z^{\tiny{(0)}}_i & \text{if} \space z^{\tiny{(1)}}_j \ge 0 \\
    0 & \text{if} \space z^{\tiny{(1)}}_j \lt 0  \\
\end{cases} =
-\sigma\bigl(\theta - G(z^{\tiny{(1)}})\bigr)2z^{\tiny{(1)}}_{j}z^{\tiny{(0)}}_i
$

$
\frac{\delta L_{neg}}{\delta w^{\tiny{(1)}}_{i, j}} = 
\frac{\delta L_{neg}}{\delta G(\bar{z}^{\tiny{(1)}})}
\frac{\delta G(\bar{z}^{\tiny{(1)}})}{\delta \bar{z}^{\tiny{(1)}}_{j}}
\frac{\delta \bar{z}^{\tiny{(1)}}_{i}}{\delta w^{\tiny{(1)}}_{i,j}}
$

$
\frac{\delta L_{neg}}{\delta G(\bar{z}^{\tiny{(1)}})} =
\sigma\bigl(G(\bar{z}^{\tiny{(1)}}) - \theta\bigr)
$

$
\frac{\delta L_{neg}}{\delta w^{\tiny{(1)}}_{i, j}} =
\begin{cases}
    \sigma\bigl(G(\bar{z}^{\tiny{(1)}}) - \theta\bigr)2\bar{z}^{\tiny{(1)}}_{j}\bar{z}^{\tiny{(0)}}_i & \text{if} \space \bar{z}^{\tiny{(1)}}_j \ge 0 \\
    0 & \text{if} \space \bar{z}^{\tiny{(1)}}_j \lt 0  \\
\end{cases} =
\sigma\bigl(G(\bar{z}^{\tiny{(1)}}) - \theta\bigr)2z^{\tiny{(1)}}_{j}\bar{z}^{\tiny{(0)}}_i
$


Updating input layers weights:

$
w^{\tiny{(0)}}_{i, j} =
w^{\tiny{(0)}}_{i, j} -\alpha\frac{\delta L}{\delta w^{\tiny{(0)}}_{i, j}} =
w^{\tiny{(0)}}_{i, j} - 
\alpha \Bigl(\frac{\delta L_{pos}}{\delta w^{\tiny{(0)}}_{i, j}} + \frac{\delta L_{neg}}{\delta w^{\tiny{(0)}}_{i, j}}\Bigr)
$

$
\frac{\delta L_{pos}}{\delta w^{\tiny{(0)}}_{i, j}} =
\frac{\delta L_{pos}}{\delta G(z^{\tiny{(1)}})}
\frac{\delta G(z^{\tiny{(1)}})}{\delta z^{\tiny{(1)}}_{j??}}
\frac{\delta z^{\tiny{(1)}}_{j??}}{\delta h^{\tiny{(1)}}}
$
