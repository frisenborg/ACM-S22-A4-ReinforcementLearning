data {
  int<lower=1> trials;
  real init_value;
  array[trials] int<lower=-1, upper=1> feedback;
  array[trials] int<lower=1,  upper=2> choice;
  array[trials] int<lower=1,  upper=2> condition;
  real          condition1_alpha_mu;
  real<lower=0> condition1_alpha_sigma;
  real          condition2_alpha_mu;
  real<lower=0> condition2_alpha_sigma;
  real          temperature_mu;
  real<lower=0> temperature_sigma;
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
  target += normal_lpdf(condition1_alpha | condition1_alpha_mu, condition1_alpha_sigma);
  target += normal_lpdf(condition2_alpha | condition2_alpha_mu, condition2_alpha_sigma);
  target += normal_lpdf(temperature | temperature_mu, temperature_sigma);
  
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

