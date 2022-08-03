---
title: "Unlock the Black Box : A Complete Guide to Explanable AI and Model Interpretability"
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

I won't deep dive into the mathematics of marginal contributions but for now, just think of Marginal Contributions as **wighted average of the playoff gains** of each player who participates in the game. Playoff gains here means the reward or penalty in some cases for each of the players.

Let's take up another example to understand the significance of Shapley Values in a better and simpler manner. Consider a model that we are building that has to decide whether to disperse a loan to a person based on a number of features. A particular retiree John Doe finds out that his loan application was rejected with a 70% chance of him not able to clear the dues an defaulting on the amount. Jphn asks the bank manager as to why he was denied the loan whereas his friend Josh another retiree was given the loan. It's quite obvious that technical nuances of the model isn't going to be comprehended well by John and we need an altrnate method.

TLDR, Here is the query that John is facing : **Why was I denied the loan with a certainity of default being 70%, whereas the avg percentage of denial for applicants in the same range was 20%?**

> Shapley values helps us here by taking the output of the model on John along with a 'comparative' group/team of other applicants and calculates the exact difference in parameters between John and the other players/loan applicants.

Let's consider three factors that will contribute towards rejection or acceptance of a proposal for loan : 
1. Credit Card Debt
2. Low Net Worth
3. Low Income Post Retirement

If John's default rate is 60% and the default rate of accepted applicants is 20%, we will have to explain the difference of 40% with a comparative assignment to each of the above three features.
Shapley algorithm might assign 25% to credit card debt, 10% to low net worth and 5% to low income post retirement. We are calculating the **average marginal contribution** of each of the features that could have contributed to the rejection of the proposal. 

I'm confident you have understood the nuance of marginal contributions now! Here's a quick summary of Shapley values.

<img src = "/blog/explanable-ai/shapleysumm.PNG">

### What are the advantages of Shapley values?

- Efficient : Shapley values are amongst the most efficient explanable AI methods where the total sum of the marginal contributions of each entity(features) fed as input to the model must add up to the total gain(model score).

- Equality : Players(features) not participating in the game(not contributing to the overall model score) should be assigned a shapley value of zero.

