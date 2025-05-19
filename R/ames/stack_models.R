# Model stacking (Ames Housing)

library(tidyverse)
library(rsample)
library(ranger)
library(catboost)
library(AmesHousing)
library(this.path)

setwd(dirname(this.path()))

ames <- make_ames() |> rename(y = Sale_Price)

seed <- 42

set.seed(seed)

split <- initial_split(ames, prop = 0.7)

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

stack_tst <- tibble(z1 = y_hat_tst_cb, z2 = y_hat_tst_rf, y = tst$y)

saveRDS(stack_tst, file = "stack_tst_ames.RDS")

# stacking the models

begin <- Sys.time()

stack_trn <- tibble()

K <- 10 # number of folds
fold <- sample(1:K, size = nrow(trn), replace = TRUE)

pb <- txtProgressBar(min = 1, max = K, style = 3)

for (k in 1:K) {
    out_of_sample <- trn[fold == k, ]
    in_sample <- trn[fold != k, ]
    
    # CatBoost
    in_pool <- catboost.load_pool(data = in_sample |> select(-y), label = in_sample$y)
    cb <- catboost.train(in_pool, params = list(random_seed = seed, verbose = 0))
    out_pool <- catboost.load_pool(data = out_of_sample)
    
    # Random Forest
    rf <- ranger(y ~ ., data = in_sample, num.trees = 10^3)
    
    # base-learners predictions become features for the meta-learner
    stack_trn <- rbind(
        stack_trn, 
        tibble(
             z1 = catboost.predict(cb, out_pool),
             z2 = predict(rf, data = out_of_sample)$predictions,
             y = out_of_sample$y
        )
    )
    
    setTxtProgressBar(pb, k)
}

close(pb)

print(Sys.time() - begin) # ~ 9 minutes

saveRDS(stack_trn, file = "stack_trn_ames.RDS")
