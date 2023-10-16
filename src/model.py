import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gamma
from src.utils import compute_rf_locations, compute_rf_size, gaussian

class V1Model:
    def __init__(self,n_cortical = 72) -> None:
        x_cortical = np.linspace(0, 40, n_cortical)
        y_cortical = np.linspace(-18, 18, n_cortical)

        x_cortical, y_cortical = np.meshgrid(x_cortical, y_cortical)
        cortical_locations = (x_cortical, y_cortical)
        mu_x, mu_y = compute_rf_locations(cortical_locations)

        biological_idx = (mu_x>=0)
        
        self.mu_x, self.mu_y = -mu_x[biological_idx], mu_y[biological_idx]
        _, self.sigma = compute_rf_size((self.mu_x, self.mu_y))

        self.n_neurons = self.mu_x.shape[0]

    def compute_rf(self, x, y):
        n_pixels = x.shape[0]
        self.W = np.zeros((self.n_neurons, n_pixels))

        for i in range(self.n_neurons):
            w = gaussian(x, y, self.mu_x[i], self.mu_y[i], self.sigma[i])
            self.W[i] = w / np.sum(w)

    def compute_response(self, stimulus):
        num_timepoints = stimulus.shape[1] 
        timepoints = np.arange(num_timepoints + 32)

        if not hasattr(self, 'hrf'):
            self.hrf = self.two_gamma(timepoints)

        neuronal_response = np.dot(self.W, stimulus)
        neuronal_response = np.append(neuronal_response, np.zeros((self.n_neurons, 32)), axis=1)
        hemodynamic_response = self.convolve_hrf(neuronal_response, num_timepoints)
        return hemodynamic_response
    
    def convolve_hrf(self, neuronal_response, num_timepoints):
        hrf_fft = fft(self.hrf)
        neuronal_response_fft = fft(neuronal_response, axis=1)
        hemodynamic_response_fft = neuronal_response_fft * hrf_fft
        hemodynamic_response = ifft(hemodynamic_response_fft, axis=1).real
        return hemodynamic_response[:, :num_timepoints]

    def two_gamma(self,timepoints):
        hrf = (6 * (timepoints**5) * np.exp(-timepoints)) / gamma(6) \
                - 1 / 6 * (16 * (timepoints ** 15) * np.exp(-timepoints)) \
                    / gamma(16)
        return hrf


