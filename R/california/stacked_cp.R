# Stacked conformal prediction (California Housing)

library(tidyverse)
library(rsample)
library(ranger)
library(catboost)
library(AmesHousing)
library(this.path)

setwd(dirname(this.path()))

source("functions.R")

###

stack_tst <- readRDS(file = "stack_tst_california.RDS")

stack_trn <- readRDS(file = "stack_trn_california.RDS")

###

meta_learner <- lm(y ~ ., data = stack_trn)

y_hat_tst <- predict(meta_learner, newdata = stack_tst)

sqrt(mean((stack_tst$y - y_hat_tst)^2))

Z <- model.matrix(meta_learner)

Z_tst <- model.matrix(y ~ ., data = stack_tst)

alpha <- 0.1

system.time(
    cp_interval <- full_cp_interval(Z, stack_trn$y, Z_tst, alpha)
)

summary(cp_interval[, 2] - cp_interval[, 1])

mean(cp_interval[, 1] <= stack_tst$y & stack_tst$y <= cp_interval[, 2])
