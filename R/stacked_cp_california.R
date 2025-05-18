# Stacked conformal prediction (California Housing)

library(tidyverse)
library(rsample)
library(ranger)
library(catboost)

source("functions.R")

california <- read_csv("california.csv", show_col_types = FALSE) |>
    mutate(ocean_proximity = as_factor(ocean_proximity)) |>
    rename(y = median_house_value)

seed <- 42

set.seed(seed)

split <- initial_split(california, prop = 0.7)

trn <- training(split)
tst <- testing(split)

# Random Forest

rf <- ranger(y ~ ., data = trn, num.trees = 10^3)
y_hat_tst_rf <- predict(rf, data = tst)$predictions
sqrt(mean((tst$y - y_hat_tst_rf)^2))

# Catboost

trn_pool <- catboost.load_pool(data = trn |> select(-y), label = trn$y) 
cb <- catboost.train(trn_pool, params = list(random_seed = seed, verbose = 100)) # 0 = silent
tst_pool <- catboost.load_pool(data = tst)
y_hat_tst_cb <- catboost.predict(cb, tst_pool)
sqrt(mean((tst$y - y_hat_tst_cb)^2))

pred_tst <- tibble(z1 = y_hat_tst_cb, z2 = y_hat_tst_rf, y = tst$y)

# saveRDS(pred_tst, file = "pred_tst_california.RDS")
# pred_tst <- readRDS(file = "pred_tst_california.RDS")

# stacking the models

begin <- Sys.time()

pred <- tibble()
num_folds <- 10
fold <- sample(1:num_folds, size = nrow(trn), replace = TRUE)
pb <- txtProgressBar(min = 1, max = num_folds, style = 3)
for (k in 1:num_folds) {
    out_of_sample <- trn[fold == k, ]
    in_sample <- trn[fold != k, ]
    # CatBoost
    in_pool <- catboost.load_pool(data = in_sample |> select(-y), label = in_sample$y)
    cb <- catboost.train(in_pool, params = list(random_seed = seed, verbose = 0))
    out_pool <- catboost.load_pool(data = out_of_sample)
    # Random Forest
    rf <- ranger(y ~ ., data = in_sample, num.trees = 10^3)
    # base-models predictions become regressors for the meta-learner
    pred <- rbind(pred, tibble(z1 = catboost.predict(cb, out_pool),
                               z2 = predict(rf, data = out_of_sample)$predictions,
                               y = out_of_sample$y))
    setTxtProgressBar(pb, k)
}
close(pb)

print(Sys.time() - begin) # ~ 6 minutes

# saveRDS(pred, file = "pred_california.RDS")
# pred <- readRDS(file = "pred_california.RDS")

meta_learner <- lm(y ~ z1 + z2, data = pred)

y_hat_tst <- predict(meta_learner, newdata = pred_tst)

sqrt(mean((pred_tst$y - y_hat_tst)^2))

###

Z <- model.matrix(meta_learner)

Z_tst <- model.matrix(y ~ z1 + z2, data = pred_tst)

alpha <- 0.1

cp_interval <- full_cp_interval(Z, pred$y, Z_tst, alpha)

# saveRDS(cp_interval, file = "cp_interval_california.RDS")
# cp_interval <- readRDS(file = "cp_interval_california.RDS")

summary(cp_interval[, 2] - cp_interval[, 1])

mean(cp_interval[, 1] <= pred_tst$y & pred_tst$y <= cp_interval[, 2])
