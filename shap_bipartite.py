from scipy.special import comb, binom
from sklearn import linear_model
import itertools
import random

class shap_bipartite():
    def __init__(self, model, D, N, num_samples):
        self.model = model
        # self.data = data
        # self.D = len(self.data[0])
        # self.N = len(self.data)
        self.D = D
        self.N = N
        self.num_samples = num_samples
        self.samples = []
        self.weights = []
        self._weights = dict()
        self._shap = None

        # Generate samples
        self.generate_samples()
    
    def _get_mask(self, mask_inds):
        mask = [0.] * self.D
        for ind in mask_inds:
            mask[ind] = 1.
        return mask
    
    def _get_weights(self, mask_inds):
        mask_len = len(mask_inds)
        if mask_len not in self._weights:
            # Weights from paper. Using approximate combinations method using binom
            self._weights[mask_len] = (self.D-1)/(binom(self.D, mask_len)*mask_len*(self.D-mask_len))
        return self._weights[mask_len]

    def _update(self, mask_inds):
        self.samples.append(self._get_mask(mask_inds))
        self.weights.append(self._get_weights(mask_inds))

    def generate_samples(self):
        while len(self.samples) < self.num_samples:
            elem = random.randrange(1, 2 ** self.D)
            mask = []
            # Convert number to binary
            j = 1
            for i in range(self.D):
                mask.append((elem // j) %2)
                j *= 2
            count = sum(mask)
            if count not in self._weights:
                self._weights[count] = (self.D-1)/(binom(self.D, count)*count*(self.D-count))
            self.weights.append(self._weights[count])
            self.samples.append(mask)

    def generate_samples_old(self):
        # We first add sets of size 1 and D,
        r = 1 
        remaining_samples = self.num_samples
        ind_elems = list(range(self.D))
        while len(self.samples) < self.num_samples:
            if self.D % 2 == 0 and  r == int(self.D/2):
                # Middle element
                if remaining_samples >= int(comb(self.D, r, exact=True)):
                    # Samples are greater than 2^D i.e. all masks
                    for mask_inds in itertools.combinations(ind_elems, r):
                        self._update(mask_inds)
                        remaining_samples -= 1
                else:
                    for mask_inds in itertools.combinations(ind_elems, r):
                        if remaining_samples <= 0:
                            break
                        self._update(mask_inds)
                        remaining_samples -= 1
            elif 2 * comb(self.D, r, exact=True) <= remaining_samples:
                for mask_inds in itertools.combinations(ind_elems, r):
                    self._update(mask_inds)
                    remaining_samples -= 1
                for mask_inds in itertools.combinations(ind_elems, self.D - r):
                    self._update(mask_inds)
                    remaining_samples -= 1
            else:
                for mask_inds in itertools.combinations(ind_elems, r):
                    if remaining_samples <= 0:
                        break
                    self._update(mask_inds)
                    remaining_samples -= 1
                for mask_inds in itertools.combinations(ind_elems, self.D - r):
                    if remaining_samples <= 0:
                        break
                    self._update(mask_inds)
                    remaining_samples -= 1

    def solve(self):

        # If already solved return value
        if self._shap is not None:
            return self._shap

        # Obtain the result of the function for each mask
        results = []

        # self.model : mask -> real
        for samp in self.samples:
            results.append(self.model(samp))
        
        # Use linear regression
        regr = linear_model.LinearRegression()
        regr.fit(self.samples, results, self.weights)

        # The coefficients of the linear regression line correspond to the 
        # Shapley values
        self._shap = regr.coef_
        return regr.coef_