---
layout:     post
title:      Formal Language Theory - Regular Language
date:       2015-10-20 11:45:00
author:     Hoang NT
url:        \math\formal-language-theory.html
summary:    Introduction to Formal Language Theory. As taught in Fall 2015 Fundamental of Mathematical for Computer Science class by Professor Toshio Endo, Tokyo Institute of Techonology. This post is from the 2nd lecture of the class.
categories: Mathematics 
thumbnail: math 
tags:
 - formal language
 - theory
 - regular language
 - introduction
 - computer science
 - automata
 - DFA
 - NFA
 - Derivatives
use_math: true
---

### How to describle a computer program/algorithm in general?

In a computer's world, everything is a string of bits (for now). Therefore, a simple way to think about all operations is to use a string of *symbols* and some structured rules apply on the string (which makes a *language*). When I first started learning about this topic in class, I was confused by many mathematics notations and new terms. I think one of the reason is because most of the classes start with talking about alphabet, regular language, and defines regular language using inductive definition first!. Thanks to [Introduction to the Theory of
Computation][1] by professor Michael Sipser, which define regular language based on **automata**, I am able to understand the material.

### The journey starts with intitial state!

Intuitively, an automaton (pl. automata) is an abstract *machine* that have states and its current state changes according to its previous state and input. Let's take the vending machine as an example. At the beginning, the machine stays in the *waiting* state. When you put some money in (some input!), the machine change to *counting_money* state, and if the money is sufficient (state change does not need input), the machine will change to *serving* state. Finally, it will
return to *waiting* state for the next customer. To be used in formal proof, an automaton is defined as follow: 
> 

$$a^2+b^2=c^2+\mathsf{Data = PCs}$$

