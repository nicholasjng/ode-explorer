This is the ode-explorer Python package,
a small library designed for solving, fast prototyping and visualization
of systems of ordinary differential equations (ODEs).


# Installation

This project is listed on PyPI, the Python Package Index. To obtain it via PyPI, run
```
pip install ode-explorer
```

It is very much advised to do this inside of a virtual environment to avoid bloating your 
system's own Python installation. A popular option for working with virtual environments is 
[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).


# Quickstart with Examples

For a very quick introduction to the main functionalities of this package, check out the ``ode_explorer.examples`` folder. More examples will be gradually added - if you have a suggestion, or you want to contribute your own, feel free to send me a message!

Some of the examples require a Jupyter installation. To install Jupyter Notebook or Jupyter Lab, run the following inside your created virtual environment:
```
pip install notebook    ## <---- for Jupyter Notebook
pip install jupyterlab  ## <---- for Jupyter Lab
```
It may also be required to install matplotlib for visualization, which can be done by running ``pip install matplotlib``.


# Introduction and main functionalities

## Models

Many processes in nature like radioactive decay, chemical reactions or classical mechanics can be characterized by **ordinary differential equations (ODEs).** Solving these equations for a process then directly gives a prediction of its evolution.

The number of equations that actually have closed form solutions available is actually a small minority; hence, numerical methods need to be developed to simulate more complex processes with correspondingly complex equations.

An ordinary differential equation is usually written in literature as
```
y' = f(t, y),
```

where ``f(t,y)`` is called the *model*. Based on that intuition, ode-explorer exposes the ``ODEModel`` class, which is a small wrapper around a standard Python callable with signature

```
f(t: float, y: float or np.ndarray, **kwargs)
```
where you can add special parameters for your model like reaction constants, decay rates etc. via Python's kwargs paradigm.

## Integrators and step functions

Solving ordinary differential equations in the computer happens by numerical integration. A popular method of solving ODEs are the *single-step methods*, which also encompass Runge-Kutta methods among others.

ode-explorer handles numerical integration by exposing an *Integrator* object. It has some internal state that facilitates logging among other things, and exposes two main integration APIs, ``integrate_const`` and ``integrate_adaptively``. The former can be used to integrate an ODE using a fixed step size h, while the latter can be equipped with a step size controlling mechanism, which chooses a step size based on local error estimates. For more information, check out the [textbook by Hairer, Wanner and Nørsett](https://www.springer.com/de/book/9783540566700).

**Step functions** are used to advance models in time during numerical integration. These methods usually differ in computational complexity and order of consistency; as a rule of thumb, a more accurate solution requires more computational work (as one might expect).

ode-explorer provides a ``StepFunction`` Interface that is built exactly for this purpose. Adding your own step functions is very simple - it requires only one of the following:
1. Subclass the ``StepFunction`` base class and override its ``forward`` method to calculate the estimate.
2. Initialize one of the template classes in ``ode_explorer.stepfunctions.templates`` with your chosen arguments.

Since most step functions originate from families of methods (e.g. explicit/implicit RK methods, linear multi-step methods), they can be templated rather well - templates for some of the most common step function families are given in ``ode_explorer.stepfunctions.templates``. 


## Callbacks and metrics

The main design emphasis of this library is that you can heavily customize your experiments to your liking. Two of the main instruments for this are callbacks and metrics.

*Callbacks* are designed to hook into the control flow of the numerical integration; ode-explorer exposes a ``Callback`` interface which is basically a callable with state. This concept may be familiar to users of ML libraries of scikit-learn and Tensorflow, which were the main inspiration behind this. You can do many things with callbacks, like logging, broadcasting your solver's intermittent results via websocket, check for NaN values - this is where your creativity comes in!

The same applies to *metrics* (with the corresponding ``Metric`` interface), which are also callables that can be used to compute quantities of interest after each step. Possible use cases include distance to a known ODE solution for sanity checking a step function, logging accepted and rejected steps in a step size control setting, or tracking of a first integral in a Hamiltonian system - again, the possibilities are really vast, so try it out!

## Step size control

Step size control is something like an art form - you can use the built-in ``StepSizeController`` interface to build your own. 

# Testing

Testing is still a work in progress, but will be added gradually.

# Planned features for upcoming releases

Some more feature plans that are in the mix for this library:

* Visualizations, Dashboard
* GPU support using JAX / XLA
* More builtin callbacks / metrics
* Boundary value problems (BVPs)
* Differential-Algebraic Equations (DAEs)
* Run caching / re-use, warm starting
