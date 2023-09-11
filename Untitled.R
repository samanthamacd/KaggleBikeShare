## Bike Share EDA Code 

## Libraries 
library(tidyverse) 
library(vroom)
library(DataExplorer)
library(patchwork)
library(ggplot2)

# 1. Read in the Data 

bike <- vroom("train.csv")

# 2. Perform an EDA and identify key features of the dataset 

variable_plot <- DataExplorer::plot_intro(bike) # visualization of glimpse! USE - 1 

correlation_plot <- DataExplorer::plot_correlation(bike) # correlation heat map USE - 2 

casual_plot <- casual_plot <- ggplot(data = bike, aes(x=temp, y=casual)) +  
  geom_point() + 
  geom_smooth()   

season_plot <- ggplot(data=bike, aes(x=as.factor(season), y=count)) + geom_boxplot(fill="violet"


plots <- (variable_plot + correlation_plot) / (casual_plot + casual_plot) 




