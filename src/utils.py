import numpy as np

def compute_rf_locations(cortical_locations, k = 15.0, a = 0.7, b = 80, alpha = 0.9):
    x_cortical, y_cortical = cortical_locations
    w = x_cortical + y_cortical * 1j

    t = np.exp((w + k * np.log(a / b)) / k)
    z = (a - b * t) / (t - 1)
    z = np.abs(z) * np.exp(1j / alpha * np.angle(z))

    x, y = np.real(z), np.imag(z)
    return x, y

def compute_rf_size(rf_location):
    x, y = rf_location
    eccentricity = np.sqrt(x**2 + y**2)
    diameter = np.maximum(0.172 * eccentricity - 0.25, 1)
    sigma = diameter / 4.0
    return diameter, sigma

def gaussian(x, y, mu_x, mu_y, sigma):
    return np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))

parameters = {
    'center': (4.95, 4.95),
    'width': 0.17,
    'steps': 18,
    'repetitions': 1,
    'pre_rest': 0.5,
    'post_rest': 0.5,
}

def create_stimulus(x,y,parameters):
    center = parameters['center']
    width = parameters['width']
    steps = parameters['steps']
    repetitions = parameters['repetitions']
    pre_rest = parameters['pre_rest']
    post_rest = parameters['post_rest']

    x = x - center[0]
    y = y - center[1]
    
    timepoints = pre_rest + steps * repetitions + post_rest

    stimulus = np.zeros((x.shape[0], timepoints))
    for i in range(repetitions):
        for j in range(steps):
            inner_border = j * width
            outer_border = (j + 1) * width
            horizontal = (x > inner_border) & (x <= outer_border)
            vertical = (y > inner_border) & (y <= outer_border)
            stimulus[:, int(pre_rest + i * steps + j)] = np.logical_and(horizontal, vertical)
    
    return stimulus

