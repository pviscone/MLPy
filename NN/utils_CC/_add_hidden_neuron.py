import numpy as np
from neuron_CC import Neuron
from numba import njit

def add_hidden_neuron(self, labels, n_candidate = 50, max_epoch = 500):
    transfer_data_shape = self.transfer_line.shape
    out_net = self.out_net
    E = out_net - labels
    delta_E = E - np.mean(E, axis = 0)

    def generate_neuron():
        return Neuron(data_shape = transfer_data_shape,
                      func = self.hidden_actv_f, 
                      w_init = self.w_init)

    def compute_S(labels, candidate):
        S = 0
        for k in range(self.num_output):
            o_p = candidate.out(self.transfer_line)
            E_pk = out_net[:,k] - labels[:,k]
            meanE_pk = np.mean(E_pk)
            delta_E = E_pk - meanE_pk
            S_k = np.abs(np.sum((o_p - np.mean(out_net))*delta_E))
            S += S_k
        return S

    def grad_desc_step(labels, candidate):
        w_grad, b_grad = gradient(labels, candidate, delta_E, out_net)
        candidate.weight += self.eta*w_grad
        candidate.bias   += self.eta*b_grad
        return w_grad, b_grad
    
    def quick_prop_step(labels, candidate, w_g_prev, b_g_prev):
        w_grad, b_grad = gradient(labels, candidate, delta_E, out_net)
        dw_prev = self.eta * w_g_prev
        db_prev = self.eta * b_g_prev
        if (w_g_prev - w_grad != 0).all() and (b_g_prev - b_grad != 0):
            dw = dw_prev * w_grad / (w_g_prev - w_grad)
            db = db_prev * b_grad / (b_g_prev - b_grad)
        else:
            dw = dw_prev
            db = db_prev
        candidate.weight += dw#_grad
        candidate.bias   += db#_grad
        return w_grad, b_grad

    def gradient(labels, candidate, delta_E, out_net):
        o_p = candidate.out(self.transfer_line)
        dnet = candidate.der_func(candidate.net(self.transfer_line))
        weight = candidate.weight
        params = [self.transfer_line, self.num_output, o_p, dnet, out_net, delta_E, weight, labels]
        return jit_gradient(*params)

    candidate = generate_neuron()
    best_neu = candidate
    best_S = 0
    for i in range(n_candidate):
        epoch = 0
        w_g_prev, b_g_prev = grad_desc_step(labels, candidate)
        S_start = compute_S(labels, candidate)
        S_prev = S_start
        while (epoch < max_epoch) or (S_incr > 1/100):
            w_g_prev, b_g_prev = grad_desc_step(labels, candidate)
            S = compute_S(labels, candidate)
            S_incr = np.abs((S - S_prev)/(S-S_start))
            S_prev = S
            epoch += 1
        if S > best_S:
            best_S = S
            best_neu = candidate
        candidate = generate_neuron()

    self.hid_neurons.append(best_neu)

    out_new_neu = self.hid_neurons[-1].out(self.transfer_line)

    n_data = transfer_data_shape[0]
    self.transfer_line = np.column_stack((self.transfer_line, out_new_neu))
    for out in self.out_neurons:
        out.weight = np.append(out.weight, np.random.uniform(-self.w_init, self.w_init))
    self.num_hidden += 1

@njit(cache = True, fastmath = True)
def jit_gradient(transfer_line, num_output, o_p, dnet, out_net, delta_E, weight, labels):
    w_grad = np.zeros(weight.shape)
    b_grad = 0
    for k in range(num_output):
        S_k = np.sum((o_p - np.mean(out_net))*delta_E[:,k])
        delta = np.sign(S_k) * delta_E[:,k] * dnet
        delta_per_input = np.empty(transfer_line.shape)
        for i in range(len(delta)):
            delta_per_input[i] = delta[i]*transfer_line[i]
        w_grad += np.sum(delta_per_input, axis=0)
        b_grad += np.sum(delta) 
    return w_grad, b_grad
