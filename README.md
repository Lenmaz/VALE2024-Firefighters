# Firefighters

Code associated with submission 7 of VALE Track of VECOMP 2024 Workshop at ECAI 2024: "Computing Value-Aligned Protocols by means of Multi-Objective Reinforcement Learning" by Manel Rodriguez-Soto, Nardine Osman, and Jordi Sabater-Mir.

Firefighters Multi-Objective Markov Decision Process environment and associated implementation of Pareto-optimal protocol computation algorithm. This environment represents a situation in which firefighters must consider two values:
- Proximity
- Professionalism

## Installation

You will require the following Python libraries in their most up-to-date version:

- Numpy
- SciPy
- MatPlotLib
- Gymnasium

## Files:

- constant.py: contains constants definition for the rest of file
- env.py: firefighters environment implementation made to be compatible with the Gymnasium framework.
- execution.py: evaluates the behaviour associated within the firefighters environment. Returns the amount of protocol alignment (i.e., accumulation of returns) obtained.
- main.py: presents examples on how to perform all the steps of our algorithm separately: 
  1. Computing the Pareto front
  2. Retrieving a Pareto-optimal protocol from the convex region of the Pareto front
  3. Retrieving a Pareto-optimal protocol from the non-convex region (or any region) of the Pareto front
  4. Evaluating a particular protocol
  5. (extra) Plotting the whole Pareto front
- pareto_front.py: contains the logic necessary to compute the Pareto front of an MOMDP.
- pmovi.py : contains our implementation of the PMOVI algorithm that computes the Pareto front of an MOMDP + a tracking algorithm to later retrieve a desired protocol from the Pareto front.
- scalarisation.py: contains several auxiliary functions that help retrieve a protocol from the convex part of the Pareto front.
