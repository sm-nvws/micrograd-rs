use crate::engine::Val;
use crate::engine::Value;
use rand::Rng;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Neuron {
    
    weights: Vec<Val>,
    bias: Val,
    nonlinear: bool,
    
}

impl Neuron {

   pub fn new(n_inputs: usize, nonlinear: bool) -> Neuron {

        Neuron {
            weights: (0..n_inputs).map(|_| Value::new(rand::thread_rng().gen_range(-1.0..1.0))).collect(),
            bias: Value::new(rand::thread_rng().gen_range(-1.0..1.0)),
            nonlinear: true
            
        }
        
    }

    fn call(&self, inputs: &Vec<Val>) -> Val {

        let mut start  = Value::new(0.0);

        for (i,j) in inputs.iter().enumerate() {

            
            start = start +  self.weights[i].clone() * j.clone();
        
    
        }

        start = start + self.bias.clone();

        
        return if self.nonlinear { start.activation() } else { start };

        
        
    }

    pub fn parameters(&self) -> Vec<Rc<RefCell<Value>>> {

        let mut params = self.weights.iter().map(|w| w.0.clone()).collect::<Vec<_>>();

        params.push(self.bias.0.clone());

        return params;
    }
}

pub struct Layer {
    
    neurons: Vec<Neuron>
    
}

impl Layer {
    
    pub fn new(n_inputs: usize, n_neurons: usize, nonlinear: bool) -> Layer {
        
        Layer {neurons: (0..n_neurons).map(|_| Neuron::new(n_inputs, nonlinear)).collect()}
    
    }        
    

    pub fn call(&self, inputs: &Vec<Val>) -> Vec<Val> {

         self.neurons.iter().map(|n| n.call(inputs)).collect()   
        
    }

    pub fn parameters(&self) -> Vec<Rc<RefCell<Value>>> {
    
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
        
    }
}

pub struct MLP {

    layers: Vec<Layer>
    
}

impl MLP {

    pub fn new(n_inputs: usize, layer_sizes: Vec<usize>) -> MLP {

        let n_layers = layer_sizes.len();
        let sizes: Vec<usize> = std::iter::once(n_inputs).chain(layer_sizes.into_iter()).collect();
        return MLP {layers: sizes.windows(2).enumerate().map(|(i, w)| {let nonlinear = i < n_layers - 1; Layer::new(w[0], w[1], nonlinear)}).collect()};
        
    }

    pub fn call(&self, inputs: Vec<Val>) -> Vec<Val> {

        self.layers.iter().fold(inputs, |x, layer| layer.call(&x))

        
    }

    pub fn parameters(&self) -> Vec<Rc<RefCell<Value>>> {

         self.layers.iter().flat_map(|n| n.parameters()).collect()

    }

}
