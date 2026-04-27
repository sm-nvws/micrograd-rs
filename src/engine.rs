use std::rc::Rc;
use std::cell::RefCell;
use std::ops::Add;
use std::ops::Mul;
use std::collections::HashSet;

#[derive(Clone)]
pub struct Val(pub Rc<RefCell<Value>>);

pub struct Value {

    pub data: f64,
    pub grad: f64,
    pub backward: Box<dyn Fn()>,
    pub children: Vec<Val>,
    
}

impl Value {

    pub fn new(data: f64) -> Val {
        
       Val(Rc::new(RefCell::new(Value{data: data, grad: 0.0,  backward: Box::new(|| {}), children: vec![] })))
    }
    
}



impl Add for Val {
    
    type Output = Val;


    fn add(self: Val, other: Val) -> Val {

        let a = self.0.clone();
        let b = other.0.clone();
        let out = Val(Rc::new(RefCell::new(Value {data: a.borrow().data + b.borrow().data, grad: 0.0, backward: Box::new(|| {}), children: vec![Val(a.clone()), Val(b.clone())]})));
        
        let out_rc = out.0.clone();

        out.0.borrow_mut().backward = Box::new(move || {
        
        let a_data = a.borrow().data;
        let b_data = b.borrow().data;
        let out_grad = out_rc.borrow().grad;

        a.borrow_mut().grad += out_grad;
        b.borrow_mut().grad += out_grad;
        
        });
        
        return out;
    
        
    }
}



impl Mul for Val {
    
    type Output = Val;

    fn mul(self: Val, other: Val) -> Val {


        let a = self.0.clone();
        let b = other.0.clone();
        let out = Val(Rc::new(RefCell::new(Value {data: a.borrow().data *  b.borrow().data, grad: 0.0, backward: Box::new(|| {}),children: vec![Val(a.clone()), Val(b.clone())]})));
        

        let out_rc = out.0.clone();

        out.0.borrow_mut().backward = Box::new(move || {
        let a_data = a.borrow().data;
        let b_data = b.borrow().data;
        let out_grad = out_rc.borrow().grad;
        a.borrow_mut().grad += b_data * out_grad;
        b.borrow_mut().grad += a_data * out_grad;
        });
               
        return out;
    
        
    }
}

fn build_topo(node: &Val, visited: &mut HashSet<usize>, result: &mut Vec<Rc<RefCell<Value>>>) {

    let id = Rc::as_ptr(&node.0) as usize;

    if !visited.contains(&id) {

        visited.insert(id);
        
        let children: Vec<Val> = node.0.borrow().children.iter().map(|c| Val(c.0.clone())).collect();
        
        for i in &children {

            build_topo(&i, visited, result);
        
        }

        result.push(node.0.clone()); 
        
    }
    
}

impl Val {

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut topo = Vec::new();
        build_topo(self, &mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0;

        for node in topo.iter().rev() {
            (node.borrow().backward)();
        }

    }

    pub fn activation(&self) -> Val {

        let a = self.0.clone();

        let out = Val(Rc::new(RefCell::new(Value{data: self.0.borrow().data.tanh(), grad: 0.0,  backward: Box::new(|| {}), children: vec![Val(a.clone())]})));
        let out_rc = out.0.clone();

        out.0.borrow_mut().backward = Box::new(move || {a.borrow_mut().grad += (1.0 - out_rc.borrow().data.powi(2)) * out_rc.borrow().grad;});
        
        return out;
        
    }

}

    

