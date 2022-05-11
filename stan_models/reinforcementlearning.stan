data {
  int<lower=1> trials;
  real init_value;
  array[trials] int<lower=-1, upper=1> feedback;
  array[trials] int<lower=1,  upper=2> choice;
  array[trials] int<lower=1,  upper=2> condition;
}

// Only estimate parameters used for the reinforcement learning
parameters {
  real<lower=0, upper=1> condition1_alpha;
  real<lower=0, upper=1> condition2_alpha;
  real<lower=0, upper=20> temperature;
}

model {
  real prediction_error;
  vector[2] value;
  vector[2] theta;
  // Priors
  target += uniform_lpdf(condition1_alpha | 0, 1);
  target += uniform_lpdf(condition2_alpha | 0, 1);
  target += uniform_lpdf(temperature | 0, 20);
  
  value[1] = init_value;
  value[2] = init_value;
  // Run the simulations
  for (trial in 1:trials) {
    theta = softmax(temperature * value);
    target += categorical_lpmf(choice[trial] | theta);
    
    prediction_error = feedback[trial] - value[choice[trial]];
    // Switch alpha depending on condition
    if (condition[trial] == 1)
      value[choice[trial]] = value[choice[trial]] + condition1_alpha * prediction_error;
    
    else if (condition[trial] == 2)
      value[choice[trial]] = value[choice[trial]] + condition2_alpha * prediction_error;
  }
}

