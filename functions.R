get_scores <- function(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv) {
    beta_hat_new <- beta_hat + ZtZ_new_inv %*% z0 * as.numeric(y0 - t(z0) %*% beta_hat)
    
    y_hat <- as.numeric(Z %*% beta_hat_new)
    y0_hat <- as.numeric(t(z0) %*% beta_hat_new)
    
    y_res <- abs(y - y_hat)
    y0_res <- abs(y0 - y0_hat)
    
    beta_res_hat <- ZtZ_inv %*% t(Z) %*% y_res
    beta_res_hat_new <- beta_res_hat + ZtZ_new_inv %*% z0 * as.numeric(y0_res - t(z0) %*% beta_res_hat)
    
    delta_hat <- as.numeric(Z %*% beta_res_hat_new)
    delta0_hat <- as.numeric(t(z0) %*% beta_res_hat_new)
    
    r <- y_res / (1 + delta_hat)
    r_hat <- sort(r)[ceiling((1 - alpha)*(nrow(Z) + 1))]
    
    r0 <- y0_res / (1 + delta0_hat)
    
    list(r0 = r0, r_hat = r_hat)
}

full_cp_interval <- function(Z, y, Z_tst, alpha, epsilon = 1e-2, sd_multiple = 5) {
    sd_y <- sd(y)
    
    ZtZ_inv <- solve(t(Z) %*% Z)
    beta_hat <- ZtZ_inv %*% t(Z) %*% y
    
    cp_interval <- matrix(0, nrow = nrow(Z_tst), ncol = 2)
    
    pb <- txtProgressBar(min = 1, max = nrow(Z_tst), style = 3)
    
    for (i in 1:nrow(Z_tst)) {
        z0 <- Z_tst[i, ]
        y0_guess <- as.numeric(t(z0) %*% beta_hat)
        
        z0t_ZtZ_inv <- t(z0) %*% ZtZ_inv
        ZtZ_new_inv <- ZtZ_inv - (ZtZ_inv %*% z0 %*% z0t_ZtZ_inv) / (1 + as.numeric(z0t_ZtZ_inv %*% z0))

        inf <- y0_guess
        lower <- inf - sd_multiple * sd_y
        while (inf - lower > epsilon) {
            y0 <- (lower + inf) / 2
            scores <- get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv)
            if (scores$r0 <= scores$r_hat) { inf <- y0 } else { lower <- y0 }
        }
        cp_interval[i, 1] <- inf
        
        sup <- y0_guess
        upper <- sup + sd_multiple * sd_y
        while (upper - sup > epsilon) {
            y0 <- (sup + upper) / 2
            scores <- get_scores(z0, y0, Z, y, alpha, ZtZ_inv, beta_hat, ZtZ_new_inv)
            if (scores$r0 <= scores$r_hat) { sup <- y0 } else { upper <- y0 }
        }
        cp_interval[i, 2] <- sup
        
        setTxtProgressBar(pb, i)
    }
    
    close(pb)
    
    cp_interval
}
