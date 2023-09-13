### 13 September - recipe(y~., data = mydata...
#   - the y~. specifies what the response is! (don't just say 'y'...) 

library(tidymodels)
library(tidyverse)
library(vroom)

bike <- vroom("train.csv")

# 1. Perform at least one cleaning step using dplyr 

first_clean_bikes <- bikes %>% 
  select(-'casual')

