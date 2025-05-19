# Conformalized Quantile Regression (California Housing)

library(tidyverse)
library(rsample)
library(ranger)
library(this.path)

setwd(dirname(this.path()))

california <- read_csv("california.csv", show_col_types = FALSE) |>
    mutate(ocean_proximity = as_factor(ocean_proximity)) |>
    rename(y = median_house_value)

set.seed(42)

split <- initial_split(california, prop = 0.7)

trn <- training(split)
tst <- testing(split)

idx_cal <- 1:2000

cal <- trn[idx_cal, ]
trn <- trn[-idx_cal, ]

rf <- ranger(y ~ ., data = trn, quantreg = TRUE)

alpha <- 0.1

alpha_low <- alpha / 2
alpha_high <- 1 - alpha / 2

q_hat_cal <- predict(rf, data = cal, type = "quantiles", quantiles = c(alpha_low, alpha_high))$predictions
E <- pmax(q_hat_cal[, 1] - cal$y, cal$y - q_hat_cal[, 2])
E_hat <- sort(E)[(1 - alpha)*(nrow(cal) + 1)]
q_hat_tst <- predict(rf, data = tst, type = "quantiles", quantiles = c(alpha_low, alpha_high))$predictions

lower <- q_hat_tst[, 1] - E_hat
upper <- q_hat_tst[, 2] + E_hat

summary(upper - lower)

mean(lower <= tst$y & tst$y <= upper)
