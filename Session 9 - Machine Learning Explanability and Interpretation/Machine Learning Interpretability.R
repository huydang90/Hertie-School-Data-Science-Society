#################################
# code block #1

# Helper packages
library(dplyr)      # for data wrangling
library(ggplot2)    # for awesome graphics

# Modeling packages
library(h2o)       # for interfacing with H2O
library(recipes)   # for ML recipes
library(rsample)   # for data splitting
library(xgboost)   # for fitting GBMs

# Model interpretability packages
library(pdp)       # for partial dependence plots (and ICE curves)
library(vip)       # for variable importance plots
library(iml)       # for general IML-related functions
library(DALEX)     # for general IML-related functions
library(lime)      # for local interpretable model-agnostic explanations


#################################

# Load and split the Ames housing data
ames <- AmesHousing::make_ames()
set.seed(123)  # for reproducibility
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)
h2o.init()

# Make sure we have consistent categorical levels
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Create training & test sets for h2o
train_h2o <- prep(blueprint, training = ames_train, retain = TRUE) %>%
  juice() %>%
  as.h2o()
test_h2o <- prep(blueprint, training = ames_train) %>%
  bake(new_data = ames_test) %>%
  as.h2o()

# Get response and feature names
Y <- "Sale_Price"
X <- setdiff(names(ames_train), Y)

#################################
# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)

# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 1000, mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train & cross-validate a GBM model
#best_gbm <- h2o.gbm(
#  x = X, y = Y, training_frame = train_h2o, ntrees = 5000, learn_rate = 0.01,
#  max_depth = 7, min_rows = 5, sample_rate = 0.8, nfolds = 10,
#  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
#  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
#  stopping_tolerance = 0
#)

# Train & cross-validate an XGBoost model
#best_xgb <- h2o.xgboost(
#  x = X, y = Y, training_frame = train_h2o, ntrees = 5000, learn_rate = 0.05,
#  max_depth = 3, min_rows = 3, sample_rate = 0.8, categorical_encoding = "Enum",
#  nfolds = 10, fold_assignment = "Modulo", 
#  keep_cross_validation_predictions = TRUE, seed = 123, stopping_rounds = 50,
#  stopping_metric = "RMSE", stopping_tolerance = 0
#)

#################################
# Train a stacked tree ensemble
ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "my_tree_ensemble",
  base_models = list(best_glm, best_rf),
  metalearner_algorithm = "drf"
)

#################################
# code block #2

# Compute predictions
predictions <- predict(ensemble_tree, train_h2o) %>% as.vector()

# Print the highest and lowest predicted sales price
paste("Observation", which.max(predictions), 
      "has a predicted sale price of", scales::dollar(max(predictions))) 
## [1] "Observation 1825 has a predicted sale price of $663,136"
paste("Observation", which.min(predictions), 
      "has a predicted sale price of", scales::dollar(min(predictions)))  
## [1] "Observation 139 has a predicted sale price of $47,245.45"

# Grab feature values for observations with min/max predicted sales price
high_ob <- as.data.frame(train_h2o)[which.max(predictions), ] %>% select(-Sale_Price)
low_ob  <- as.data.frame(train_h2o)[which.min(predictions), ] %>% select(-Sale_Price)


#################################
# code block #3

vip(ensemble_tree, method = "model")
## Error in vi_model.default(ensemble_tree, method = "model") : 
##  model-specific variable importance scores are currently not available for objects of class "h2o.stackedEnsemble.summary".


#################################
# code block #4

# 1) create a data frame with just the features
features <- as.data.frame(train_h2o) %>% select(-Sale_Price)

# 2) Create a vector with the actual responses
response <- as.data.frame(train_h2o) %>% pull(Sale_Price)

# 3) Create custom predict function that returns the predicted values as a vector
pred <- function(object, newdata)  {
  results <- as.vector(h2o.predict(object, as.h2o(newdata)))
  return(results)
}

# Example of prediction output
pred(ensemble_tree, features) %>% head()
## [1] 207144.3 108958.2 164248.4 241984.2 190000.7 202795.8


#################################
# code block #5

# iml model agnostic object
components_iml <- Predictor$new(
  model = ensemble_tree, 
  data = features, 
  y = response, 
  predict.fun = pred
)

# DALEX model agnostic object
components_dalex <- DALEX::explain(
  model = ensemble_tree,
  data = features,
  y = response,
  predict_function = pred
)


#################################
# code block #6

vip(
  ensemble_tree,
  train = as.data.frame(train_h2o),
  method = "permute",
  target = "Sale_Price",
  metric = "RMSE",
  nsim = 5,
  sample_frac = 0.5,
  pred_wrapper = pred
)


#################################
# code block #7

# Custom prediction function wrapper
pdp_pred <- function(object, newdata)  {
  results <- mean(as.vector(h2o.predict(object, as.h2o(newdata))))
  return(results)
}

# Compute partial dependence values
pd_values <- partial(
  ensemble_tree,
  train = as.data.frame(train_h2o), 
  pred.var = "Gr_Liv_Area",
  pred.fun = pdp_pred,
  grid.resolution = 20
)
head(pd_values)  # take a peak
##   Gr_Liv_Area     yhat
## 1         334 158858.2
## 2         584 159566.6
## 3         835 160878.2
## 4        1085 165896.7
## 5        1336 171665.9
## 6        1586 180505.1

# Partial dependence plot
autoplot(pd_values, rug = TRUE, train = as.data.frame(train_h2o))


#################################
# code block #8

# Construct c-ICE curves
partial(
  ensemble_tree,
  train = as.data.frame(train_h2o), 
  pred.var = "Gr_Liv_Area",
  pred.fun = pred,
  grid.resolution = 20,
  plot = TRUE,
  center = TRUE,
  plot.engine = "ggplot2"
)


#################################
# code block #9

interact <- Interaction$new(components_iml)

interact$results %>% 
  arrange(desc(.interaction)) %>% 
  head()
##        .feature .interaction
## 1  First_Flr_SF   0.13917718
## 2  Overall_Qual   0.11077722
## 3  Kitchen_Qual   0.10531653
## 4 Second_Flr_SF   0.10461824
## 5      Lot_Area   0.10389242
## 6   Gr_Liv_Area   0.09833997

plot(interact)


#################################
# code block #10

interact_2way <- Interaction$new(components_iml, feature = "First_Flr_SF")
interact_2way$results %>% 
  arrange(desc(.interaction)) %>% 
  top_n(10)
##                      .feature .interaction
## 1   Overall_Qual:First_Flr_SF   0.14385963
## 2     Year_Built:First_Flr_SF   0.09314573
## 3   Kitchen_Qual:First_Flr_SF   0.06567883
## 4      Bsmt_Qual:First_Flr_SF   0.06228321
## 5  Bsmt_Exposure:First_Flr_SF   0.05900530
## 6  Second_Flr_SF:First_Flr_SF   0.05747438
## 7  Kitchen_AbvGr:First_Flr_SF   0.05675684
## 8    Bsmt_Unf_SF:First_Flr_SF   0.05476509
## 9     Fireplaces:First_Flr_SF   0.05470992
## 10  Mas_Vnr_Area:First_Flr_SF   0.05439255


#################################
# code block #11

# Two-way PDP using iml
interaction_pdp <- Partial$new(
  components_iml, 
  c("First_Flr_SF", "Overall_Qual"), 
  ice = FALSE, 
  grid.size = 20
) 
plot(interaction_pdp)


#################################
# code block #12

# Create explainer object
components_lime <- lime(
  x = features,
  model = ensemble_tree, 
  n_bins = 10
)

class(components_lime)
## [1] "data_frame_explainer" "explainer"            "list"
summary(components_lime)
##                      Length Class              Mode     
## model                 1     H2ORegressionModel S4       
## preprocess            1     -none-             function 
## bin_continuous        1     -none-             logical  
## n_bins                1     -none-             numeric  
## quantile_bins         1     -none-             logical  
## use_density           1     -none-             logical  
## feature_type         80     -none-             character
## bin_cuts             80     -none-             list     
## feature_distribution 80     -none-             list


#################################
# code block #13

# Use LIME to explain previously defined instances: high_ob and low_ob
lime_explanation <- lime::explain(
  x = rbind(high_ob, low_ob), 
  explainer = components_lime, 
  n_permutations = 5000,
  dist_fun = "gower",
  kernel_width = 0.25,
  n_features = 10, 
  feature_select = "highest_weights"
)


#################################
# code block #14

glimpse(lime_explanation)
## Observations: 20
## Variables: 11
## $ model_type       <chr> "regression", "regression", "regression", "regr…
## $ case             <chr> "1", "1", "1", "1", "1", "1", "1", "1", "1", "1…
## $ model_r2         <dbl> 0.41704102, 0.41704102, 0.41704102, 0.41704102,…
## $ model_intercept  <dbl> 172912.3, 172912.3, 172912.3, 172912.3, 172912.…
## $ model_prediction <dbl> 410826.3, 410826.3, 410826.3, 410826.3, 410826.…
## $ feature          <chr> "Gr_Liv_Area", "Overall_Qual", "Total_Bsmt_SF",…
## $ feature_value    <dbl> 3627, 8, 1930, 35760, 1796, 1831, 14, 807, 1, 3…
## $ feature_weight   <dbl> 54818.866, 48655.518, 40207.059, 20089.463, 198…
## $ feature_desc     <chr> "2141 < Gr_Liv_Area", "Overall_Qual = Very_Exce…
## $ data             <list> [[Two_Story_1946_and_Newer, Residential_Low_De…
## $ prediction       <dbl> 663136.4, 663136.4, 663136.4, 663136.4, 663136.…


#################################
# code block #15

plot_features(lime_explanation, ncol = 1)


#################################
# code block #16

# Tune the LIME algorithm a bit
lime_explanation2 <- explain(
  x = rbind(high_ob, low_ob), 
  explainer = components_lime, 
  n_permutations = 5000,
  dist_fun = "euclidean",
  kernel_width = 0.75,
  n_features = 10, 
  feature_select = "lasso_path"
)

# Plot the results
plot_features(lime_explanation2, ncol = 1)


#################################
# code block #17

# Compute (approximate) Shapley values
(shapley <- Shapley$new(components_iml, x.interest = high_ob, sample.size = 1000))
## Interpretation method:  Shapley 
## Predicted value: 663136.380000, Average prediction: 181338.963590 (diff = 481797.416410)
## 
## Analysed predictor: 
## Prediction task: unknown 
## 
## 
## Analysed data:
## Sampling from data.frame with 2199 rows and 80 columns.
## 
## Head of results:
##        feature         phi      phi.var
## 1  MS_SubClass  1746.38653 4.269700e+07
## 2    MS_Zoning   -24.01968 3.640500e+06
## 3 Lot_Frontage  1104.17628 7.420201e+07
## 4     Lot_Area 15471.49017 3.994880e+08
## 5       Street     1.03684 6.198064e+03
## 6        Alley    41.81164 5.831185e+05
##                          feature.value
## 1 MS_SubClass=Two_Story_1946_and_Newer
## 2    MS_Zoning=Residential_Low_Density
## 3                     Lot_Frontage=118
## 4                       Lot_Area=35760
## 5                          Street=Pave
## 6                Alley=No_Alley_Access

# Plot results
plot(shapley)


#################################
# code block #18

# Reuse existing object
shapley$explain(x.interest = low_ob)

# Plot results
shapley$results %>%
  top_n(25, wt = abs(phi)) %>%
  ggplot(aes(phi, reorder(feature.value, phi), color = phi > 0)) +
  geom_point(show.legend = FALSE)


#################################
# code block #19

# Compute tree SHAP for a previously obtained XGBoost model
X <- readr::read_rds("data/xgb-features.rds")
xgb.fit.final <- readr::read_rds("data/xgb-fit-final.rds")


#################################
# code block #20

# Try to re-scale features (low to high)
feature_values <- X %>%
  as.data.frame() %>%
  mutate_all(scale) %>%
  gather(feature, feature_value) %>% 
  pull(feature_value)

# Compute SHAP values, wrangle a bit, compute SHAP-based importance, etc.
shap_df <- xgb.fit.final %>%
  predict(newdata = X, predcontrib = TRUE) %>%
  as.data.frame() %>%
  select(-BIAS) %>%
  gather(feature, shap_value) %>%
  mutate(feature_value = feature_values) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)))

# SHAP contribution plot
p1 <- ggplot(shap_df, aes(x = shap_value, y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 0.4, alpha = 0.25) +
  xlab("SHAP value") +
  ylab(NULL)

# SHAP importance plot
p2 <- shap_df %>% 
  select(feature, shap_importance) %>%
  filter(row_number() == 1) %>%
  ggplot(aes(x = reorder(feature, shap_importance), y = shap_importance)) +
  geom_col() +
  coord_flip() +
  xlab(NULL) +
  ylab("mean(|SHAP value|)")

# Combine plots
gridExtra::grid.arrange(p1, p2, nrow = 1)


#################################
# code block #21

shap_df %>% 
  filter(feature %in% c("Overall_Qual", "Gr_Liv_Area")) %>%
  ggplot(aes(x = feature_value, y = shap_value)) +
  geom_point(aes(color = shap_value)) +
  scale_colour_viridis_c(name = "Feature value\n(standardized)", option = "C") +
  facet_wrap(~ feature, scales = "free") +
  scale_y_continuous('Shapley value', labels = scales::comma) +
  xlab('Normalized feature value')


#################################
# code block #22

high_breakdown <- prediction_breakdown(components_dalex, observation = high_ob)

# class of prediction_breakdown output
class(high_breakdown)
## [1] "prediction_breakdown_explainer" "data.frame" 

# check out the top 10 influential variables for this observation
high_breakdown[1:10, 1:5]
##                                      variable contribution variable_name variable_value cummulative
## 1                                 (Intercept)    181338.96     Intercept              1    181338.9
## Gr_Liv_Area              + Gr_Liv_Area = 4316     46971.64   Gr_Liv_Area           4316    228310.5
## Second_Flr_SF          + Second_Flr_SF = 1872     52997.40 Second_Flr_SF           1872    281307.9
## Total_Bsmt_SF          + Total_Bsmt_SF = 2444     41339.89 Total_Bsmt_SF           2444    322647.8
## Overall_Qual  + Overall_Qual = Very_Excellent     47690.10  Overall_Qual Very_Excellent    370337.9
## First_Flr_SF            + First_Flr_SF = 2444     56780.92  First_Flr_SF           2444    427118.8
## Bsmt_Qual             + Bsmt_Qual = Excellent     49341.73     Bsmt_Qual      Excellent    476460.6
## Neighborhood      + Neighborhood = Northridge     54289.27  Neighborhood     Northridge    530749.8
## Garage_Cars                 + Garage_Cars = 3     41959.23   Garage_Cars              3    572709.1
## Kitchen_Qual       + Kitchen_Qual = Excellent     59805.57  Kitchen_Qual      Excellent    632514.6
