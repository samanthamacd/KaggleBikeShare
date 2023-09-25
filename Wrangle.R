### 13 September - recipe(y~., data = mydata...
#   - the y~. specifies what the response is! (don't just say 'y'...) 

library(tidymodels)
library(tidyverse)
library(vroom)
library(poissonreg) 

bike <- vroom("train.csv")
bike_test <- vroom("test.csv")

# 1. Perform at least one cleaning step using dplyr 

first_clean_bike <- bike %>% 
  select(-c('casual', 'registered'))

# 2. Perform at least two feature enginerring steps using recipe() 

my_recipe <- recipe(count~., data=first_clean_bike) %>% 
  step_time(datetime, features=c("hour", "minute")) %>% 
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_poly(temp, degree=2) %>% 
  step_corr(all_predictors()) 
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data=bike)

# 3. Fit a linear regression Model 

my_mod <- linear_reg() %>% 
  set_engine("lm") # Engine = what R function to use (there are many many options) 


bike_workflow <- workflow() %>% # sets up a series of steps that you can apply to any dataset 
  add_recipe(my_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data=first_clean_bike) # fit the workflow 

bike_predictions <- predict(bike_workflow, 
                            new_data = bike_test) # Use fit to predict 

# 4. Get rid of negative predictions 
bike_predictions[bike_predictions < 0] <- 0 
bike_predictions[,-1]
bike_predictions$datetime <- as.character(format(bike_test$datetime))
names(bike_predictions)[1] <- "count"
# 5. Export as a csv file 
vroom_write(bike_predictions, "BikePredictions.csv", delim=',')



# 5. Create a poisson regression model - 18 September 

pois_mod <- poisson_reg() %>% # type of model 
  set_engine("glm") # GLM = generalized linear model 

bike_pois_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data=bike) # fit the workflow 

bike_predict <- predict(bike_pois_workflow, 
                        new_data=bike_test) 

bike_predict$datetime <- as.character(format(bike_test$datetime))
names(bike_predict)[1] <- "count"
vroom_write(bike_predict, "PoissonPredictions.csv", delim = ",")


# 6 Create a Penalized Regression Model on log(count) 

# First transform to log(count)
logTrainSet <- first_clean_bike %>%
  mutate(count=log(count))

my_recipe <- recipe(count~., data=logTrainSet) %>% 
  step_time(datetime, features=c("hour", "minute")) %>% 
  step_zv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_rm(datetime) %>% 
  step_poly(temp, degree=2) %>% 
  step_poly(windspeed, degree=2) %>%
  step_poly(weather, degree = 2) %>%
  step_corr(all_predictors())
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data=bike)


preg_model <- linear_reg(penalty=1, mixture = 0) %>% # mixture = 0 has gotten me the lowest score so far
  set_engine("glmnet") 

preg_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(preg_model) %>% 
  fit(data=logTrainSet) 

log_lin_preds <- predict(preg_workflow, new_data = bike_test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(.pred, datetime) %>% #Just keep datetime and predictions
  mutate(datetime=as.character(format(datetime))) %>%
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count))

vroom_write(log_lin_preds, "PenalizedPredictions.csv", delim = ",")

