# Framerwork

Here is an overall diagram of the AO agent [@wang2024survey;@sumers2023cognitive].

<img src="../img/diagram.png" alt="diagram" width="600"/>

## Task

A text description of the task.

## Planner

It has two parts: scientific method and reinforcement learning.

Initialization:

* propose parameters to measure current situation (as state for RL), how to measure each parameter
* propose quantifiable criterion for goal accomplishment

Scientific method:

* make observation
* make hypothesis (abductive reasoning)
* what follows the hypothesis (deductive reasoning)
* test the hypothesis
* summarize to get general principle (inductive reasoning)

Reinforcement learning (RL, based on When2Ask [@hu2023enabling]):

* state (S) = current situation (parametrization based on LLM)
* action space (A) = {ask for a new plan, continue current plan}
* reward (R) = 10 if goal accomplished; 1 if in line with previous successful experience; -5 if new plan = old plan; 0 otherwise
* discount ($\gamma$) = 0.99

## Memory

There are short-term memory and long-term memory.

Short-term memory is the chat history.

Long-term memory includes:

* Procedural memory stores the production system itself: the set of rules that can be applied to working memory to determine the agent’s behavior.
* Semantic memory stores facts about the world.
* Episodic memory stores sequences of the agent’s past behaviors.

Long-term memory is stored as knowledge graph. It can provide successful experience, or use as few-shot example in prompt.
<!-- can be replaced by memory engrams if neuromorphic implementation -->

## Actor

## Explainer

## Role/Profile

researcher, programmer, paper reader... (some can be used as functions)

## Environment

## Human in the loop

## References

\bibliography

