import topk
import numpy as np


class ModelGenerator:

    def __init__(self):
        self.vectors = None
        self.d = None
        self.j = None
        self.top_k = None
        self.unWrapFunction = None
        self.evaluationFunction = None
        self.evaluationFunctions = None
        self.executed = False
        self.init_top_k = None
        self.init_top_ks = None

    def database(self, database):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        self.vectors = database
        return self

    def target(self, target):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        self.j = target
        return self

    def k(self, k):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        self.top_k = k
        return self

    def unwrap_func(self, unwrap_func):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        self.unWrapFunction = unwrap_func
        return self

    def eval_func(self, eval_func):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        self.evaluationFunction = eval_func
        return self

    def eval_funcs(self, eval_funcs):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        self.evaluationFunctions = eval_funcs
        return self

    # ---Preprocessing

    def setup_top_k(self):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        if self.vectors is None or self.evaluationFunction is None or self.top_k is None:
            raise Exception(
                "Model is missing parameters! Required parameters are database, k, and evaluation function.")

        initialTuples = topk.generateTuplesSubset(self.vectors, self.evaluationFunction,
                                                  [1 for x in range(len(self.vectors[0]))], self.unWrapFunction)
        self.init_top_k = set(topk.computeTopK(initialTuples, self.top_k))

    def setup_top_ks(self):
        if self.executed:
            raise Exception("Model cannot be modified once run!")

        if self.vectors is None or self.evaluationFunctions is None or self.top_k is None:
            raise Exception(
                "Model is missing parameters! Required parameters are database, k, and evaluation functions.")

        self.init_top_ks = set()
        for evaluationFunction in range(len(self.evaluationFunctions)):
            initialTuples = topk.generateTuplesSubset(self.vectors, self.evaluationFunctions[evaluationFunction],
                                                      [1 for x in range(len(self.vectors[0]))], self.unWrapFunction)
            if topk.computeInTopK(initialTuples, self.top_k, self.j):
                self.init_top_ks.add(evaluationFunction)

    # ----Model Methods

    def in_top_k(self, masks):
        if self.vectors is None or self.evaluationFunction is None or self.top_k is None or self.j is None:
            raise Exception("Model is missing parameters! Required parameters are database, k, target, "
                            "and evaluation function.")

        self.executed = True

        result = []
        for mask in masks:
            tuples = topk.generateTuplesSubset(self.vectors, self.evaluationFunction, mask, self.unWrapFunction)
            inTopK = topk.computeInTopK(tuples, self.top_k, self.j)
            result.append(1 if inTopK else 0)

        return np.reshape(result, len(masks, 1))

    def not_in_top_k(self, mask):
        if self.vectors is None or self.evaluationFunction is None or self.top_k is None or self.j is None:
            raise Exception("Model is missing parameters! Required parameters are database, k, target, "
                            "and evaluation function.")
        self.executed = True

        tuples = topk.generateTuplesSubset(self.vectors, self.evaluationFunction, mask, self.unWrapFunction)
        return 1 if not topk.computeInTopK(tuples, self.top_k, self.j) else 0

    def top_k_look_like_this(self, mask):
        if self.vectors is None or self.evaluationFunction is None or self.top_k is None:
            raise Exception("Model is missing parameters! Required parameters are database, k, "
                            "and evaluation function.")

        if self.init_top_k is None:
            raise Exception("Model is not setup! Run model.setup_top_k to compute initial values.")

        self.executed = True
        tuples = topk.generateTuplesSubset(self.vectors, self.evaluationFunction, mask, self.unWrapFunction)
        newTopK = set(topk.computeTopK(tuples, self.top_k))
        return 1 if (len(newTopK) == 0 and len(self.init_top_k) == 0) else len(
            newTopK.intersection(self.init_top_k)) / len(newTopK.union(self.init_top_k))

    def in_these_top_ks(self, mask):
        if self.vectors is None or self.evaluationFunctions is None or self.top_k is None:
            raise Exception("Model is missing parameters! Required parameters are database, k, "
                            "and evaluation functions.")

        if self.init_top_ks is None:
            raise Exception("Model is not setup! Run model.setup_top_ks to compute initial values.")

        self.executed = True

        top_ks = set()
        for evaluationFunction in range(len(self.evaluationFunctions)):
            tuples = topk.generateTuplesSubset(self.vectors, self.evaluationFunctions[evaluationFunction], mask,
                                               self.unWrapFunction)
            if topk.computeInTopK(tuples, self.top_k, self.j):
                top_ks.add(evaluationFunction)
        return 1 if (len(top_ks) == 0 and len(self.init_top_ks) == 0) else len(
            self.init_top_ks.intersection(top_ks)) / len(self.init_top_ks.union(top_ks))


def test():
    database = [[5, 3, 1], [2, 4, 4], [3, 1, 2], [4, 1, 3], [1, 2, 5]]
    weights = [5, 4, 1]
    evaluation_function = lambda e: (sum([(weights[x] * e[x]) if e[x] is not None else 0 for x in range(len(weights))]))

    model = ModelGenerator()
    model.database(database).eval_func(evaluation_function).k(2).target(0).setup_top_k()

    masks = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0]]

    results_in_top_k = []

    for mask in masks:
        results_in_top_k.append(model.in_top_k(mask))

    assert(results_in_top_k == [0, 1, 1, 1, 1, 1])

    results_not_in_top_k = []

    for mask in masks:
        results_not_in_top_k.append(model.not_in_top_k(mask))

    assert(results_not_in_top_k == [1, 0, 0, 0, 0, 0])

    top_k_look_like_this = []

    for mask in masks:
        top_k_look_like_this.append(model.top_k_look_like_this(mask))

    assert(top_k_look_like_this == [1/3, 1/1, 1/3, 1/1, 1/3, 1/1])

    model = ModelGenerator()

    weights1 = [80, 90, 4]
    weights2 = [90, 10, 5]
    weights3 = [50, 60, 10]

    evalFuncs = [lambda e:(sum([(weights1[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])),
                 lambda e:(sum([(weights2[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))])),
                 lambda e:(sum([(weights3[x]*e[x]) if e[x] is not None else 0 for x in range(len(weights))]))]

    model.database(database).eval_funcs(evalFuncs).k(2).target(0).setup_top_ks()

    in_these_top_ks = []

    for mask in masks:
        in_these_top_ks.append(model.in_these_top_ks(mask))

    assert(in_these_top_ks == [0/1, 1/1, 1/1, 2/3, 1/1, 1/1])
    print('All tests passed!')

if __name__ == "__main__":
    test()
