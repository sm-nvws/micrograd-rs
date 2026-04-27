mod engine;
mod nn;

use engine::Val;
use engine::Value;
use nn::MLP;


fn main() {

    let mlp = MLP::new(2, vec![4, 4, 1]);

   

    let ys = vec![1.0, -1.0, 1.0, -1.0];

    
 
    for j in 1..=200 {

        let learning_rate = Value::new(0.2);

        let inputs: Vec<Val> = vec![Value::new(2.0), Value::new(6.5)]; //  example 

        let xs = vec![
        vec![Value::new(2.0), Value::new(3.0)],
        vec![Value::new(-1.0), Value::new(-1.0)],
        vec![Value::new(0.5), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(-2.0)],];

        let mut values: Vec<Val> = vec![];


        for i in xs {

           values.push(mlp.call(i).into_iter().next().unwrap());       
        
        }

        let mut loss = Value::new(0.0);
        for (pred, target) in values.iter().zip(ys.iter()) {
        
            let diff = pred.clone() + Value::new(-*target);
            loss = loss + diff.clone() * diff.clone();
        
        }

        loss.backward();


        
        for i in mlp.parameters() {
        
            let grad = i.borrow().grad;
            i.borrow_mut().data -= 0.01 * grad;
                       
        }

       
        for i in mlp.parameters() {

                       
            i.borrow_mut().grad = 0.0;

            
        }

    }

    let test_xs = vec![
    vec![Value::new(2.0), Value::new(3.0)],
    vec![Value::new(-1.0), Value::new(-1.0)],
    vec![Value::new(0.5), Value::new(1.0)],
    vec![Value::new(1.0), Value::new(-2.0)],];


    for (i, (input, target)) in test_xs.into_iter().zip(ys.iter()).enumerate() {
        
        let pred = mlp.call(input).into_iter().next().unwrap();
        
        println!("input {}: pred = {}, target = {}", i, pred.0.borrow().data, target);
        
    }
    
}
    
    

