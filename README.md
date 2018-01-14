# Labeled variational-autoencoder
My experiments in generating MNIST using a Variational Autoencoder

This is derived from [kvfrans's blogpost about variational autoencoders.](http://kvfrans.com/variational-autoencoders-explained/)

# Hyperparameter Optimization Choice

I have chosen [chocolate|https://github.com/AIworx-Labs/chocolate]. It
supports all types of parameters, it has several algorithms (for
example, it is the only Bayesian Opt framework I've found that can
make a Pareto frontier). I like the idea of independent processes
using the database for communications of results - it should be easily
distributed.

Because I do not plan to stay on MNIST for ever, being able to deal
with a large training set is important. I did not find any
easy-to-install Python packages supporting cost-sensitive optimization
(FABOLAS from [AutoML::RoBO|http://www.ml4aad.org/automl/robo/]
supports it, but RoBO is very intrusive and difficult to install. So
others wouldn't be able to extend my results easily.) I think I should
be able to extend chocolate to cost-sensitive optimization.

* I should be easily able to add the cost as an optional field in the
  database - other algorithms just don't see it. And I can have it
  default to run-time by including a run-start and run-stop.
* I should also be able to specify parameters that should be rolled up
  for predicting the final optimized value - so one can predict
  expected improvement/upper confidence bound per time for the maximum
  cost values of those parameters. I'll want to reference the FABOLAS
  paper if I implement those.
* I can import GPy to enable MCMC inference that will eliminate the
  kernel length-scale dependence
