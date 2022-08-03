---
title: "A Complete Guide to Explanable AI and Model Interpretability"
dateString: Aug 2022
description: "An End to End Practical Guide to Model Interpretability with LIME and SHAP"

draft: false
tags: ["ML", "Python", "Projects", "Climate Change AI","Energy Projects","Explanable AI"]
weight: 102
---

# Overview

Model Interpretability and Explanable AI has become an integral part of a Data Science project that helps us understand model predictions. A machine learning model is often just a black box and it often becomees difficult to answer the question 'Why is the model predicting the label?' or 'Why should I believe you that this solution is accurate?'

## Better Interpretability Leads to Better Adoption

A sophisticated machine learning algorithm usually can produce accurate predictions, but its notorious “black box” nature does not help adoption at all. 

Think about this: If you ask me to swallow a black pill without telling me what’s in it, I certainly don’t want to swallow it. The interpretability of a model is like a label on a drug bottle. We need to make our effective pill transparent for easy adoption.

There are many challenges to deal with while working with black box ML models but on the upside there are many methods to explain, comprehend and effectively illuminate exactly how the decisions/answers that the ML model comes up with are correct.

In this article, I'm going to give you a hands on practical view of how model explanability can help explain the decisions taken in a non-model centric and agnostic manner to convert code into phrasable conclusions that can be understood by the business and management in an organization.

## What are Shapley Values?

Let's take an example to understand Shapley Values. 

In football there are many positions that a player can be fielded in : Goalkeeper, Defender, Midfield and Striker. Let's assume that in a hypothetical world, the players who contribute in scoring a goal would get a monetary reward of 50$/goal. Do you think all the players would get the same share of the money when a goal is scored?

Obviously no right. A striker must get a greater share of the reward compared to a defender, who may have passed on the ball to the striker but did not take the crucial shot. Now let's jump to a fundamental question...

**How do we decide the 'share of money' that must be given to each player?**

> Shapley Values capture the marginal contribution of each player to the end result.

I won't deep dive into the mathematics of marginal contributions but for now, just think of Marginal Contributions as **wighted average of the playoff gains** of each player who participates in the game.