### 13 September - recipe(y~., data = mydata...
#   - the y~. specifies what the response is! (don't just say 'y'...) 

library(tidymodels)
library(tidyverse)
library(vroom)
library(poissonreg) 
library(stacks)

bike <- vroom("train.csv")
bikeTest <- vroom("test.csv")

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
  set_engine("lm") 

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

# Tuning Parameters 

preg_model <- linear_reg(penalty=tune(), mixture=tune()) %>% # mixture = 0 has gotten me the lowest score so far
  set_engine("glmnet") 

preg_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(preg_model) 

tuning_grid <- grid_regular(penalty(), 
                            mixture(), 
                            levels = 10) 

folds <- vfold_cv(logTrainSet, v = 10, repeats=1)

CV_results <- preg_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid, 
            metrics = metric_set(rmse, mae, rsq)) 

collect_metrics(CV_results) %>% 
  filter(.metric =="rmse") %>% 
  ggplot(data = ., aes(x=penalty, y=mean, color=factor(mixture))) + 
  geom_line()

bestTune <- CV_results %>% 
  select_best("rmse")

final_wf <- preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data=logTrainSet) 

more_preds <- final_wf %>% 
  predict(new_data = bike_test)

more_preds$datetime <- as.character(format(bike_test$datetime))
names(more_preds)[1] <- "count"
vroom_write(more_preds, "TuningPreds.csv", delim = ",")


# Regression Trees! (with log count)

my_mod <- decision_tree(tree_depth = tune(), # setting it up to tell it later 
                        cost_complexity = tune(), 
                        min_n = tune()) %>% 
  set_engine("rpart") %>%  # what R function to use 
  set_mode("regression") 


tree_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod) 

tree_tuning_grid <- grid_regular(min_n(), 
                            tree_depth(),
                            cost_complexity(),
                            levels = 10) 

folds <- vfold_cv(logTrainSet, v = 5, repeats=1)

CV_results <- tree_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid, 
            metrics = metric_set(rmse, mae)) 

# collect_metrics(CV_results) %>% 
#   filter(.metric =="rmse") %>% 
#   ggplot(data = ., aes(x=penalty, y=mean, color=factor(mixture))) + 
#   geom_line()

bestTune <- CV_results %>% 
  select_best("rmse")

final_tree_wf <- tree_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data=logTrainSet) 

more_preds <- final_tree_wf %>% 
  predict(new_data = bike_test)

more_preds$datetime <- as.character(format(bike_test$datetime))
names(more_preds)[1] <- "count"
more_preds$count <- exp(more_preds$count)
vroom_write(more_preds, "TreePreds.csv", delim = ",")

# Random Forests! 

random_mod <- rand_forest(mtry = tune(), 
                          min_n = tune(), 
                          trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression") 


forest_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(random_mod) 

tuning_grid <- grid_regular(min_n(), 
                            mtry(range=c(1,10))) 

folds <- vfold_cv(logTrainSet, v = 5, repeats=1)

CV_results <- forest_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid, 
            metrics = metric_set(rmse, mae)) 

bestTune <- CV_results %>% 
  select_best("rmse")

final_forest_wf <- forest_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data=logTrainSet) 

more_preds <- final_forest_wf %>% 
  predict(new_data = bike_test)

more_preds$datetime <- as.character(format(bike_test$datetime))
names(more_preds)[1] <- "count"
more_preds$count <- exp(more_preds$count)
vroom_write(more_preds, "ForestPreds.csv", delim = ",")



# Stacking Models 

folds <- vfold_cv(logTrainSet, v=10)

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples() 

# Linear Regression - candidate #1 
linreg_folds_fit <- bike_workflow %>% 
  fit_resamples(resamples = folds,
                metric = metric,
                control = tunedModel) 

# Penalized Regression - candidate #2  
preg_model <- linear_reg(penalty=tune(), mixture=tune()) %>% # mixture = 0 has gotten me the lowest score so far
  set_engine("glmnet") 

preg_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(preg_model) 

tuning_grid <- grid_regular(penalty(), 
                            mixture(), 
                            levels = 10) 
preg_models <- preg_wf %>% 
  tune_grid(resamples=folds, 
            grid=tuning_grid, 
            metrics = metric_set(rmse, mae), 
            control = untunedModel) 

# Regression Tree - candidate #3 
tree_mod <- decision_tree(tree_depth = tune(), # setting it up to tell it later 
                        cost_complexity = tune(), 
                        min_n = tune()) %>% 
  set_engine("rpart") %>%  # what R function to use 
  set_mode("regression") 


tree_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(tree_mod) 

tree_models <- tree_wf %>% 
  tune_grid(resamples = folds, 
            grid=tree_tuning_grid, 
            metrics = metric_set(rmse, mae), 
            control = untunedModel) 

bike_stack <- stacks() %>% 
  add_candidates(linreg_folds_fit) %>% 
  add_candidates(preg_models) %>% 
  add_candidates(tree_models)
as_tibble(bike_stack) 

fitted_stack <- bike_stack %>% 
  blend_predictions() %>% 
  fit_members() 

collect_parameters(fitted_stack, "tree_folds_fit") # which were kept from trees

stacked_preds <- predict(fitted_stack, new_data=bikeTest) %>% 
  mutate(.pred=exp(.pred)) %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(.pred, datetime) %>% #Just keep datetime and predictions
  mutate(datetime=as.character(format(datetime))) %>%
  rename(count=.pred)
  
vroom_write(stacked_preds, "StackedPreds.csv", delim = ",")


