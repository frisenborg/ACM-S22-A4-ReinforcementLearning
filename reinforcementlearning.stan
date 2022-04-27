

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> trials;
  vector[trials] feedback;
  vector[trials] choice;
  vector[trials] condition;
  real initValue;
  
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real<lower=0, upper=1> alpha_1;
  real<lower=0, upper=1> alpha_2;
  real<lower=0, upper=20> temperature;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  real pe;
  vector[2] value;
  vector[2] theta; 
  
  target += uniform_lpdf(alpha_1 | 0, 1);
  target += uniform_lpdf(alpha_2 | 0, 1);
  target += uniform_lpdf(temperature | 0, 20);
  
  value = initValue;
  
  for (trial in 1:trials) {
    theta = softmax(temperature * value);
    target += categorical_lpmf(choice[trial] | theta);
    
    pe = feedback[trial] - value[choice[trial]];
    
    value[choice[trial]] = (alpha_1 * (1-condition[trial]) + alpha_2 * (condition[trial])) * pe
  }
}

