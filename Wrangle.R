### 13 September - recipe(y~., data = mydata...
#   - the y~. specifies what the response is! (don't just say 'y'...) 

library(tidymodels)
library(tidyverse)
library(vroom)

bike <- vroom("train.csv")

# 1. Perform at least one cleaning step using dplyr 

first_clean_bike <- bike %>% 
  select(-c('casual', 'registered')) 
view(first_clean_bike)

# 2. Perform at least two feature enginerring steps using recipe() 

my_recipe <- recipe(count~., data=bike) %>% 
  step_date(datetime, features = "dow") %>% 
  step_time(datetime, features=c("hour", "minute")) %>% 
  step_zv(all_predictors()) %>% 
  step_select(-c("casual", "registered")) 
prepped_recipe <- prep(my_recipe)
view(bake(prepped_recipe, new_data=bike))
  