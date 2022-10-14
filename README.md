# Explaining Top-K Matching

## Shapley Values for Explanation in Two-sided Matching Applications

## Abstract
In this paper, we initiate research in explaining matchings. In particular, we consider the large-scale two-sided matching applications where preferences of the users are specified as (ranking) functions over a set of attributes and matching recommendations are derived as top-k. We consider multiple natural explanation questions, concerning the users of these systems. Observing the competitive nature of these environments, we propose multiple Shapley-based approaches for explanation. Besides exact algorithms, we propose a sampling-based approximation algorithm with provable guarantees to overcome the combinatorial complexity of the exact Shapley computation. Our extensive experiments on real-world and synthetic data sets validate the usefulness of our proposal and confirm the efficiency and accuracy of our algorithms.

## Publication to Cite

Paper currently under review.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine, and allow you to rerun the experiments from the paper.

* [Python 3.10](https://www.python.org/downloads/)

## Using the Queries and Methods

First import the experiments module. The parameters are defined as follow.

* tuples - an array of n arrays of d integers
* evalFunc - a function on array of d integers where inputs can potentially be "None" producing a positive value
* evalFuncs - if a query requires more than one evalFunc, they are passed in as an array of functions
* k - the number of individuals in a given top-k
* j - the target tuple index on which to perform the query "Why was $t_j$ in the top k of the evaluator?"
* d - the total number of shapley values to compute
* unWrapFunction - assuming the shapley values are being calculated over groups of attributes, this function takes an array of group indexes, and returns the full list of associated attributes
* m - for the approximate method, this is the number of permutations sampled. note that it will evaluate once for each index in the permutation
* bruteForce - if bruteForce has been run on the dataset, include bruteForce to receive the Standard Deviation and Average Difference between the methods

Additionally, the SHAP methods will use a model parameter. To generate a model, first import ModelGenerator from model.py, and then call the following builder methods.

* database - the same as tuples before
* eval_func - the same as evalFunc before
* eval_funcs - the same as evalFuncs before
* k - the same as k before
* target - the same as j before
* attributes - the same as d before
* unwrap_func - the same as unWrapFunc before

Finally, once all data has been given to the model, the following methods must be called to prepare the data for specific queries:

* For SQ-Single, call the function "setup_top_k" to construct the top-k for the queried individual's evaluation function.
* For SQ-Multiple, call the function "setup_top_ks" to construct the top-k for all individuals' evaluation functions.

## Running the Experiments

* Real World Dataset - Varying M on Brute Force, Approximate, and SHAP. Call `python3 candidates_set_experiment.py`
* Synthetic Dataset - Varying M on Brute Force, Approximate, and SHAP. Call `python3 varying_m_experiment.py [DISTRIBUTIONS]` where distributions are space separated values in AZL (Anti-correlated Zipfian Linear), CZL (Correlated Zipfian Linear), IZL (Independent Zipfian Linear), AZNL (Anti-correlated Zipfian Non-Linear), CZNL (Correlated Zipfian Non-Linear), IZNL (Independent Zipfian Non-Linear).
* Synthetic Dataset - Varying D on Brute Force, Approximate, and SHAP. Call `python3 varying_d_experiment.py [DISTRIBUTIONS]` where distributions are space separated values in AZL (Anti-correlated Zipfian Linear), CZL (Correlated Zipfian Linear), IZL (Independent Zipfian Linear), AZNL (Anti-correlated Zipfian Non-Linear), CZNL (Correlated Zipfian Non-Linear), IZNL (Independent Zipfian Non-Linear).
* Case Study - Call `python3 case_study_experiment.py`
* Top Attribute Experiment - Determining how often the top value of the method is the same as brute force for Approximate, Weights, and Attribute Score. Call `python3 top_attribute_experiment.py [DISTRIBUTIONS]` where distributions are space separated values in AZL (Anti-correlated Zipfian Linear), CZL (Correlated Zipfian Linear), IZL (Independent Zipfian Linear), AZNL (Anti-correlated Zipfian Non-Linear), CZNL (Correlated Zipfian Non-Linear), IZNL (Independent Zipfian Non-Linear), .
