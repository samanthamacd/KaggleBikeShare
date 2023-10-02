# 2 October - Model Stacking 

# Random Forest - first introduction to 'ensemble methods' 
#   - ensemble methods: uses a collection of models to make a better prediction
#     - bagging: bootstrap sample then average preds from each bootstrap to reduce variance 
#     - boosting: learns incrementally about the data from previous fits of the same model to reduce bias 
#     - stacking: combines predictions of diverse models to reduct bias and uses a meta-learner that make the final preds 
# 
# What do I use when? 
#   - stacking: fit all the models, then combine all of those predictions into a single prediction 
# Stacking: 
#   - diverse base learners are each able to learn a part of the data = combined they know more of the data 
#   - base learners (1,2,3,4, etc.) make preds -> all feed into a meta learner = final prediction 
#       - the 'explanatory variables' in the meta learner are the preds from each base 
#       - base learners should capture different parts of the data 
#   - you can stack existing models, or you can throw variations on 'untuned' models 
#   - 
