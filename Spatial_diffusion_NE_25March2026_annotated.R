########################
# Spatial Diffusion - Balázs Lengyel, Northeastern 25 March 2026
#
# ANNOTATED VERSION
# -----------------
# This script contains four main blocks:
#   1. Scaling and gravity calculations over the technology life cycle
#   2. Bass diffusion estimation for the whole country
#   3. Bass diffusion estimation for all towns and comparison of empirical vs. estimated peaks
#   4. A toy agent-based diffusion simulation on a sample social network
#
# The annotations below explain, step by step, what each block is doing,
# what data are loaded, what transformations are applied, what models are estimated,
# and what figures are produced.
########################

# Paper can be found at https://www.nature.com/articles/s41598-020-72137-w


########################
# Housekeeping
########################

# Remove every object from the current R environment to ensure that the script
# starts from a clean state and no leftover objects influence the results.
rm(list = ls())

# Set the working directory to the folder containing the input data files and
# where the output figures will be saved.
# NOTE: This path is machine-specific and may need to be changed on another computer.
setwd("c:\\Users\\lengyel.balazs\\Desktop\\LengyelB_2022June\\Northeastern teaching\\coding class")



######################################
# 1. Scaling and gravity
######################################

########################
# 1.1 Scaling over the life cycle
########################

# Read individual-level adoption category data.
# "id_adopter.csv" contains a user identifier and an adoption category.
# "id_cityid.csv" maps each user identifier to a city identifier.
# "cityid_pop_poplog_2557.csv" contains city population and its logarithm for 2,557 towns.
id_adopter <- read.table("id_adopter.csv", sep = ",", header = TRUE)
id_cityid  <- read.table("id_cityid.csv", sep = ",", header = TRUE)
pop        <- read.table("cityid_pop_poplog_2557.csv", sep = ";", header = TRUE)

# Merge user-to-city data with adopter category data so that each user now has:
#   - an id
#   - a cityid
#   - an adopter category
id_city_adopt <- merge(id_cityid, id_adopter, by = "id")

# Create a counter variable n=1 so that simple summation later counts users.
id_city_adopt$n <- 1

# Collapse the original adopter categories into 3 broader life-cycle stages:
#   ad3 = 1  -> innovators
#   ad3 = 2  -> early adopters
#   ad3 = 3  -> majority and laggards together
id_city_adopt$ad3 <- 1
id_city_adopt$ad3[id_city_adopt$adopter == 2] <- 2
id_city_adopt$ad3[id_city_adopt$adopter == 3 |
                    id_city_adopt$adopter == 4 |
                    id_city_adopt$adopter == 5] <- 3

# Count how many adopters fall into each aggregated adoption stage within each city.
city_adopt <- aggregate(id_city_adopt$n,
                        by = list(id_city_adopt$cityid, id_city_adopt$ad3),
                        FUN = sum)
names(city_adopt) <- c("cityid", "ad3", "N")

# Add population information to each city-stage observation.
city_adopt <- merge(city_adopt, pop, by = "cityid")

# Remove large intermediate objects that are no longer needed.
rm(id_adopter, id_cityid, id_city_adopt, pop)

# Restrict the analysis to cities above 10,000 inhabitants.
# This removes very small settlements where scaling may be unstable.
c_a <- city_adopt[city_adopt$pop > 10000, ]

# Compute the base-10 logarithm of the number of adopters,
# to estimate standard urban scaling regressions in log-log form.
c_a$N_log <- log10(c_a$N)

# Estimate separate scaling regressions for the 3 adoption stages:
#   log10(number of adopters) ~ log10(population)
ad1 <- lm(N_log ~ pop_log, data = c_a[c_a$ad3 == 1, ])
summary(ad1)   # Reported slope: 1.41, SE = 0.09

ad2 <- lm(N_log ~ pop_log, data = c_a[c_a$ad3 == 2, ])
summary(ad2)   # Reported slope: 1.28, SE = 0.04

ad3 <- lm(N_log ~ pop_log, data = c_a[c_a$ad3 == 3, ])
summary(ad3)   # Reported slope: 1.07, SE = 0.01

# Compute confidence intervals for the regression coefficients.
confint(ad1)
confint(ad2)
confint(ad3)

# Generate fitted values and confidence bands for plotting.
ci_ad1 <- predict(ad1, interval = "confidence", level = 0.95)
ci_ad2 <- predict(ad2, interval = "confidence", level = 0.95)
ci_ad3 <- predict(ad3, interval = "confidence", level = 0.95)

# Bind the fitted values back to the corresponding city observations.
plot_1 <- cbind(c_a[c_a$ad3 == 1, ], ci_ad1)
plot_2 <- cbind(c_a[c_a$ad3 == 2, ], ci_ad2)
plot_3 <- cbind(c_a[c_a$ad3 == 3, ], ci_ad3)

# The scales package is used below to create transparent confidence polygons.
library(scales)

# Create a PNG figure showing:
#   - city observations by adoption stage
#   - fitted scaling lines
#   - 95% confidence ribbons
png("fig1_scaling.png", width = 600, height = 600)

# Start the plot with innovators.
plot(plot_1$pop_log,
     plot_1$N_log,
     type = "p", col = "red", pch = 2, cex = 2.5,
     ylab = "", xlab = "", yaxt = "n", xaxt = "n", main = "",
     xlim = c(4, 6.5), ylim = c(0, 6))

# Add fitted line and confidence interval lines for innovators.
lines(plot_1$pop_log, plot_1$fit, type = "l", col = "red", lty = 1, cex = 2.5)
lines(plot_1$pop_log, plot_1$lwr, type = "l", col = "red", lty = 2, pch = 3, cex = 2.5)
lines(plot_1$pop_log, plot_1$upr, type = "l", col = "red", lty = 2, pch = 3, cex = 2.5)

# Add points, fitted line, and confidence interval lines for early adopters.
lines(plot_2$pop_log, plot_2$N_log, type = "p", col = "green", pch = 3, cex = 2.5)
lines(plot_2$pop_log, plot_2$fit, type = "l", col = "green", lty = 1, cex = 2.5)
lines(plot_2$pop_log, plot_2$lwr, type = "l", col = "green", lty = 2, pch = 3, cex = 2.5)
lines(plot_2$pop_log, plot_2$upr, type = "l", col = "green", lty = 2, pch = 3, cex = 2.5)

# Add points, fitted line, and confidence interval lines for majority & laggards.
lines(plot_3$pop_log, plot_3$N_log, type = "p", col = "blue", pch = 4, cex = 2.5)
lines(plot_3$pop_log, plot_3$fit, type = "l", col = "blue", lty = 1, cex = 2.5)
lines(plot_3$pop_log, plot_3$lwr, type = "l", col = "blue", lty = 2, pch = 3, cex = 2.5)
lines(plot_3$pop_log, plot_3$upr, type = "l", col = "blue", lty = 2, pch = 3, cex = 2.5)

# Add semi-transparent confidence ribbons for each group.
polygon(c(plot_1$pop_log, rev(plot_1$pop_log)),
        c(plot_1$upr, rev(plot_1$lwr)),
        col = alpha("red", 0.3), border = NA)
polygon(c(plot_2$pop_log, rev(plot_2$pop_log)),
        c(plot_2$upr, rev(plot_2$lwr)),
        col = alpha("green", 0.3), border = NA)
polygon(c(plot_3$pop_log, rev(plot_3$pop_log)),
        c(plot_3$upr, rev(plot_3$lwr)),
        col = alpha("blue", 0.3), border = NA)

# Custom axes are shown with powers of ten so the reader sees the original scale.
axis(1, font = 2, cex.axis = 2, col.axis = "black", las = 1,
     at = c(4, 4.5, 5, 5.5, 6, 6.5),
     labels = c(expression(10^4), expression(10^4.5), expression(10^5),
                expression(10^5.5), expression(10^6), expression(10^6.5)),
     col = "black", title = "Population")

axis(2, font = 2, cex.axis = 2, col.axis = "black", las = 2,
     at = c(0, 1, 2, 3, 4, 5),
     labels = c(expression(10^0), expression(10^1), expression(10^2),
                expression(10^3), expression(10^4), expression(10^5)),
     col = "black", title = "Adopters")

# Add legend and text labels with the estimated scaling coefficients and standard errors.
legend(4.3, 1,
       legend = c("Innovators", "Early adopters", "Majority & Laggards"),
       text.col = c("red", "green", "blue"),
       lty = c(1, 1), col = c("red", "green", "blue"),
       lwd = c(3, 3), cex = 1.5,
       box.lwd = 0, box.col = "white", bg = NULL)

text(5.8, 0.65, expression(beta == 1.41), cex = 1.5, col = "red")
text(5.8, 0.35, expression(beta == 1.28), cex = 1.5, col = "green")
text(5.8, 0.05, expression(beta == 1.07), cex = 1.5, col = "blue")

text(6.2, 0.65, ", SE=0.09", cex = 1.5, col = "red")
text(6.2, 0.35, ", SE=0.04", cex = 1.5, col = "green")
text(6.2, 0.05, ", SE=0.01", cex = 1.5, col = "blue")

dev.off()



########################
# 1.2 Gravity over the life cycle
########################

# Clear the environment to start this subsection independently.
rm(list = ls())

# Read precomputed gravity-group data.
# This file appears to contain binned distance values ("group_gravity")
# and log-probabilities of invitation/adoption links for different adopter groups.
inv_grav <- read.table("inv_gravity.csv", sep = ",", header = TRUE)

# Inspect the first rows to understand the structure.
head(inv_grav)

# Compute R-squared values by squaring Pearson correlations between
# log distance and log invitation probability for each adoption group.
(cor(inv_grav$P_d_1_log,   inv_grav$group_gravity_log,
     method = "pearson", use = "complete.obs"))^2   # 0.24
(cor(inv_grav$P_d_2_log,   inv_grav$group_gravity_log,
     method = "pearson", use = "complete.obs"))^2   # 0.85
(cor(inv_grav$P_d_345_log, inv_grav$group_gravity_log,
     method = "pearson", use = "complete.obs"))^2   # 0.92

# Produce a figure comparing invitation probabilities over distance
# for innovators, early adopters, and majority/laggards.
png(filename = "fig1_inv_grav.png", width = 600, height = 600, units = "px", bg = "white")

# Plot innovators first, restricting the distance range to 5 < gravity < 305.
# The x-axis uses logged distance, the y-axis uses logged probabilities.
plot(inv_grav$group_gravity_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
     inv_grav$P_d_1_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
     type = "b", col = "red",
     cex = 2, pch = 16, cex.lab = 1,
     ylab = "Probability of Invitations",
     xlab = "Distance", axes = FALSE, frame.plot = TRUE,
     ylim = c(-7.8, -4.8), xlim = c(1, 2.5))

# Add the curves for early adopters and majority/laggards.
lines(inv_grav$group_gravity_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      inv_grav$P_d_2_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      type = "b", col = "green",
      cex = 2, pch = 16, cex.lab = 1)

lines(inv_grav$group_gravity_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      inv_grav$P_d_345_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      type = "b", col = "blue",
      cex = 2, pch = 16, cex.lab = 1)

# Add three reference power-law lines with different exponents.
# These act as visual guides for how steeply invitation probabilities decline with distance.
lines(inv_grav$group_gravity_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      inv_grav$line1[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      type = "l", col = "black",
      cex = 2, lty = "dotted", lwd = 2)

lines(inv_grav$group_gravity_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      inv_grav$line2[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      type = "l", col = "black",
      cex = 2, lty = "dashed", lwd = 2)

lines(inv_grav$group_gravity_log[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      inv_grav$line3[inv_grav$group_gravity > 5 & inv_grav$group_gravity < 305],
      type = "l", col = "black",
      cex = 2, lty = "solid", lwd = 2)

# Custom axis labels display original values in powers of ten.
axis(side = 1, at = c(1, 1.5, 2, 2.5),
     labels = c(expression(10^1), expression(10^1.5),
                expression(10^2), expression(10^2.5)),
     cex.axis = 2)

axis(side = 2, at = c(-8, -7, -6, -5),
     labels = c(expression(10^-8), expression(10^-7),
                expression(10^-6), expression(10^-5)),
     cex.axis = 2, mgp = c(4, 1, 0))

# Add a legend for the three reference slopes.
legend(1, -7,
       legend = c(expression(italic(d)^-0.3),
                  expression(italic(d)^-0.7),
                  expression(italic(d)^-1.1)),
       lty = c("dotted", "dashed", "solid"), lwd = 2,
       col = "black", cex = 2,
       bty = "n", x.intersp = 0.5)

dev.off()



###########################################
# 2. Bass DE estimation
###########################################

# Clear the environment again to isolate this section.
rm(list = ls())

# Read monthly country-level adoption data for fitting a Bass diffusion model.
c <- read.table("BASS_curve_country.csv", sep = ";", header = TRUE)

########################
# 2.1 Bass estimation for the whole country
########################

# Define the time variable (months).
ctime <- c$month

# First, compute an externally parameterized Bass cumulative adoption curve
# using fixed P and Q values taken from outside the current estimation.
P <- 0.000223
Q <- 0.094

ngete <- exp(-(P + Q) * (ctime))
Bcdf  <- ((1 - ngete) / (1 + (Q / P) * ngete))

# Estimate the Bass model from the country-level monthly registration rates.
# creg = new adopters per month
# ccumreg = cumulative adopters
creg    <- c$reg_rate_month
ccumreg <- cumsum(creg)

# Fit a nonlinear least squares version of the Bass hazard / density.
# 0.31311578 appears to be the market potential M used as a scaling factor.
cnls <- nls(creg ~ (0.31311578 * (((cP + cQ)^2 / cP) *
                                    exp(-(cP + cQ) * (ctime)) /
                                    (1 + (cQ / cP) * exp(-(cP + cQ) * (ctime)))^2)),
            start = list(cP = 0.00696, cQ = 0.0964))

# Extract estimated coefficients.
cBcoef <- coef(cnls)
cp <- cBcoef[1]
cq <- cBcoef[2]

# Construct the fitted cumulative Bass curve from estimated parameters.
cngete <- exp(-(cp + cq) * (ctime))
cBcdf  <- (0.31311578 * (1 - cngete) / (1 + (cq / cp) * cngete))

# Store estimated p and q again under simpler variable names.
Bcoef <- coef(cnls)
p <- Bcoef[1]
q <- Bcoef[2]

# Inspect the estimated Bass model and its confidence intervals.
summary(cnls)
confint(cnls)

# Smooth observed monthly registrations with a 5-month moving average.
library(TTR)
c$smooth <- as.numeric(SMA(c$reg_month_country, n = 5))

# Convert the externally parameterized Bass cumulative curve into monthly increments.
Bcdf_m <- numeric(length = length(Bcdf))
Bcdf_m[1] <- Bcdf[1]
Bcdf_m[2] <- Bcdf[2] - Bcdf[1]
for (i in 3:127) {
  Bcdf_m[i] <- Bcdf[i] - Bcdf[i - 1]
}

# Construct a second Bass cumulative curve using estimated parameters.
# NOTE: The original code keeps "ngete" from the external P,Q in the numerator,
# while "ngete2" is used in the denominator. This may be intentional for comparison,
# but it is unusual and worth checking if exact Bass consistency is required.
ngete2 <- exp(-(p + q) * (ctime))
Bcdf2  <- ((1 - ngete) / (1 + (q / p) * ngete2))

# Convert the estimated cumulative curve into monthly increments.
Bcdf_m2 <- numeric(length = length(Bcdf))
Bcdf_m2[1] <- Bcdf2[1]
Bcdf_m2[2] <- Bcdf2[2] - Bcdf2[1]
for (i in 3:127) {
  Bcdf_m2[i] <- Bcdf2[i] - Bcdf2[i - 1]
}

# Quick exploratory plot in the console comparing external and estimated monthly curves.
plot(ctime, Bcdf_m2, type = "l", col = "blue", lwd = 3, lty = "solid")
points(ctime, Bcdf_m, type = "l", col = "red", lwd = 3, lty = "solid")

# Create the main country-level Bass figure.
png(filename = "fig2_Bass_global_Balazs.png", width = 700, height = 500, units = "px")

par(mar = c(5.1, 7.1, 4.1, 2.1))

# Plot observed new users per month.
plot(ctime, c$reg_month_country, type = "b", col = "orange", lwd = 2,
     ylab = "", xlab = "months", cex.lab = 2, pch = 1, cex.axis = 1.5, yaxt = "n")

# Add smoothed observed series.
points(ctime[5:122], c$smooth[5:122], type = "l", col = "blue", lwd = 2)

# Add the Bass-estimated monthly adoption series rescaled by the total number of new users.
points(ctime, Bcdf_m2 * sum(c$reg_month_country),
       type = "l", col = "red", lwd = 3, lty = "solid")

# Mark the estimated peak and the smoothed empirical peak.
abline(v = ctime[which(Bcdf_m2 == max(Bcdf_m2))], lwd = 3, lty = "dashed", col = "red")
abline(v = ctime[which(c$smooth[5:122] == max(c$smooth[5:122])) + 4], lwd = 3, lty = "dotted", col = "blue")

# Legend for the three series.
legend("topright", c("New users", "Smoothed Adoption", "Estimated Adoption"),
       fill = 'white', border = 'white', cex = 1.8, pt.cex = 2,
       col = c("orange", "blue", "red"), pch = c(1, NA, NA),
       lty = c(NA, "solid", "solid"), lwd = c(NA, 2, 2), pt.bg = 'white', bty = "n")

# Visual annotation for prediction error: the horizontal distance between smoothed and estimated peaks.
arrows(10, 85000, 20, 85000, col = "black", lwd = 3)
arrows(20, 85000, 10, 85000, col = "black", lwd = 3)
segments(10, 75000, 10, 95000, col = "blue", lwd = 3, lty = "dotted")
segments(20, 75000, 20, 95000, col = "red",  lwd = 3, lty = "dashed")
text(15, 70000, "Prediction Error", cex = 1.8)
text(26, 100000, "Est. \nPeak", cex = 1.8)
text(4, 100000,  "Smo. \nPeak", cex = 1.8)

# Custom y-axis labels.
axis(2, at = c(0, 20000, 40000, 60000, 80000, 100000),
     labels = c("0",
                expression(paste("2?", 10^"4")),
                expression(paste("4?", 10^"4")),
                expression(paste("6?", 10^"4")),
                expression(paste("8?", 10^"4")),
                expression(10^5)),
     las = 2, cex.axis = 1.5)

dev.off()



##################################
# 3. Estimate Bass DE by towns
##################################

rm(list = ls())

########################
# 3.1 Bass estimation for all 2,557 towns
########################

# Read town-level Bass curve data and city attributes.
e      <- read.table("BASS_curve_towns_2557.csv", sep = ";", header = TRUE)
cities <- read.table("cityid_pop_poplog_2557.csv", sep = ";", header = TRUE)

# Use minpack.lm for more robust nonlinear least squares estimation (nlsLM).
require(minpack.lm)

# Preallocate vectors to store estimated town-level parameters and diagnostics.
qa  <- numeric(length = nrow(cities))   # imitation/adoption parameter Q
pa  <- numeric(length = nrow(cities))   # innovation/adoption parameter P
qd  <- numeric(length = nrow(cities))   # disadoption growth parameter q
xd  <- numeric(length = nrow(cities))   # disadoption initial level x
SSa <- numeric(length = nrow(cities))   # sum of squared residuals for adoption fit
SSd <- numeric(length = nrow(cities))   # sum of squared residuals for disadoption fit
mt  <- numeric(length = nrow(cities))   # market size proxy m = max cumulative adoption

# Loop over all towns and estimate a simple exponential disadoption curve.
for (i in 1:2557) {
  sel  <- e$cityid_new == i
  town <- e[sel, ]
  time <- town$month

  disad    <- town$disadoption_rate_month
  cumdisad <- cumsum(disad)

  # Starting value for x is the minimum cumulative disadoption.
  xstart <- min(cumdisad)

  # Fit cumulative disadoption as x*(1+q)^time.
  nls_d <- nlsLM(cumdisad ~ (x * (1 + q)^time),
                 start = list(x = 0.000001, q = 0.1),
                 control = list(maxiter = 500), alg = 'port')

  Dcoef <- coef(nls_d)
  xd[i]  <- Dcoef[1]
  qd[i]  <- Dcoef[2]
  SSd[i] <- sum(resid(nls_d)^2)
}

# Loop again over all towns and fit a Bass adoption curve to monthly adoption rates.
for (i in 1:2557) {
  sel  <- e$cityid_new == i
  town <- e[sel, ]
  time <- town$month

  adopt    <- town$adoption_rate_month
  cumadopt <- cumsum(adopt)

  # Set the market size m to the observed maximum cumulative adoption in the town.
  m <- max(cumadopt)

  # Fit the Bass density formula.
  nls_a <- nlsLM(adopt ~ (m * (((P + Q)^2 / P) *
                                 exp(-(P + Q) * (time)) /
                                 (1 + (Q / P) * exp(-(P + Q) * (time)))^2)),
                 start = list(P = 0.0000696, Q = 0.1),
                 control = list(maxiter = 500))

  Acoef <- coef(nls_a)
  pa[i]  <- Acoef[1]
  qa[i]  <- Acoef[2]
  SSa[i] <- sum(resid(nls_a)^2)
  mt[i]  <- m
}

# Use a short moving average to estimate the empirical timing of the adoption peak.
library("TTR")

# Store the empirical peak month based on a 3-month moving average.
T_emp3 <- numeric(length = nrow(cities))

for (i in 1:2557) {
  sel       <- e$cityid_new == i
  smoothing <- e[sel, ]
  smoothing$peak <- SMA(smoothing$adoption_rate_month, n = 3)

  sel2 <- max(smoothing$peak, na.rm = TRUE) == smoothing$peak
  peak <- smoothing[sel2, ]

  # The original code takes peak$month[3], presumably because the SMA alignment
  # creates multiple rows and the author wants the centered month.
  T_emp3[i] <- peak$month[3]
}

emp_peak <- data.frame(unique(e$cityid), T_emp3)
names(emp_peak) <- c("cityid", "T_emp3")

# Load additional datasets to simplify merging of town-level attributes.
data_pop  <- read.csv("population.csv", sep = ",", header = TRUE)
data_pop$pop  <- as.numeric(as.integer(data_pop$pop))
data_pop$KSHK <- as.numeric(data_pop$KSHK)

data_peak <- read.csv("T_adoption_peak.csv", sep = ",", header = TRUE)
data_pq   <- read.csv("qa_qd_data.csv", sep = ",", header = TRUE)
data_code <- read.csv("city_codes_pop.csv", sep = ",", header = TRUE)

# Merge code, peak, and parameter data into one town-level table.
data_code <- merge(data_code, data_peak, by.x = "cityid", by.y = "cityid")
names(data_code)[5] <- paste("peak")

data_OSN <- merge(data_code, data_pq, by.x = "cityid", by.y = "cityid")

# Remove several columns that are not needed downstream.
data_OSN <- data_OSN[, -c(10, 11, 12, 13, 14, 15)]
data_OSN <- data_OSN[, -c(14, 15, 16, 17)]

# Ensure numeric types are correct after the merge.
data_OSN$cityid   <- as.numeric(data_OSN$cityid)
data_OSN$pop_ext  <- as.numeric(data_OSN$pop_ext)
data_OSN$ksh_code <- as.numeric(data_OSN$ksh_code)
data_OSN$pop      <- as.numeric(data_OSN$pop)

# Rename selected columns to adoption/disadoption parameter names.
names(data_OSN)[11] <- paste("qa")
names(data_OSN)[12] <- paste("qd")
names(data_OSN)[13] <- paste("xd")

rm(data_code, data_peak, data_pq)



#######################
# 3.2 Empirical versus theoretical peak
#######################

# Compute the Bass-implied peak time for each town from estimated p and q:
# Tpq = -log(p/q) / (p + q)
data_OSN$Tpq <- -log(data_OSN$pa / data_OSN$qa) / (data_OSN$pa + data_OSN$qa)

# Merge in the empirically observed peak time from the smoothed data.
data_OSN <- merge(data_OSN, emp_peak, by.x = "cityid", by.y = "cityid")

# Libraries for density-based scatter plots and color palettes.
require(KernSmooth)
library(RColorBrewer)

my.cols     <- rev(brewer.pal(11, "RdYlBu"))
Lab.palette <- colorRampPalette(c("white", "red"), space = "Lab")

# Custom hook function for adding a horizontal kernel-density legend
# to the smoothScatter plot below.
fudgeit <- function() {
  xm <- get('xm',   envir = parent.frame(1))
  ym <- get('ym',   envir = parent.frame(1))
  z  <- get('dens', envir = parent.frame(1))
  colramp <- get('colramp', parent.frame(1))

  fields::image.plot(xm, ym, z, col = colramp(256), legend.only = TRUE,
                     add = FALSE, horizontal = TRUE, legend.shrink = 1,
                     smallplot = c(.183, .935, .9, .925),
                     legend.width = 5,
                     legend.lab = "Kernel density",
                     legend.cex = 2,
                     axis.args = list(cex.axis = 2),
                     legend.line = -3)
}

# Plot estimated peak month versus empirical peak month.
png("TBass_Temp_smooth_b.png", width = 750, height = 875)

par(mar = c(5, 7, 10, 3), xaxs = 'i', yaxs = 'i', bty = "n")

smoothScatter(data_OSN$Tpq, data_OSN$T_emp3, nrpoints = 0,
              colramp = colorRampPalette(c("black", "#202020", "#736AFF", "cyan",
                                           "yellow", "#F87431", "#FF7F00", "red", "#7E2217")),
              pch = 19, cex = 1, col = "blue",
              xlim = c(40, 80), ylim = c(40, 80),
              xaxt = 'n', yaxt = 'n', ann = FALSE,
              xlab = NULL, ylab = NULL,
              transformation = function(x) x^.25, axes = FALSE,
              postPlotHook = fudgeit)

# 45-degree line: perfect match between empirical and estimated peak.
abline(0, 1, col = "white", lty = 1, lwd = 3)

# Add axes and labels.
axis(side = 1, pos = 40.6, at = c(45, 55, 65, 75),
     labels = c("45", "55", "65", "75"), cex.axis = 2)
axis(side = 2, pos = 40.8, at = c(45, 55, 65, 75),
     labels = c("45", "55", "65", "75"), cex.axis = 2)
title(xlab = "Estimated peak (month)", ylab = "Empirical peak (month)", cex.lab = 3.5)

dev.off()



#######################
# 3.3 Q versus Peak with fixed P
#######################

# Generate theoretical peak curves for a range of fixed P values and observed Q values.
data_OSN$T1 <- as.numeric(-log(0.5e-4 / data_OSN$qa) / (0.5e-4 + data_OSN$qa))
data_OSN$T2 <- as.numeric(-log(1e-4   / data_OSN$qa) / (1e-4   + data_OSN$qa))
data_OSN$T3 <- as.numeric(-log(3e-4   / data_OSN$qa) / (3e-4   + data_OSN$qa))
data_OSN$T4 <- as.numeric(-log(5e-4   / data_OSN$qa) / (5e-4   + data_OSN$qa))

# Create a plotting table with estimated p and q and merged town-level attributes.
p_q_plot <- data.frame(cities$cityid, cities$cityid_new, pa, qa)

p_q_plot <- merge(p_q_plot, data_OSN,
                  by.x = "cities.cityid", by.y = "cityid", all.x = FALSE)

# Restrict to the first 1000 towns (likely for readability or sample selection).
sel <- p_q_plot$cityid_new < 1001
p_q_plot <- p_q_plot[sel, ]

# Define a custom palette for plotting p-values as colors.
myPalette <- colorRampPalette(c("red", "orange", "#1affcc", "blue"), space = "rgb")

# Build breaks for P values to inform the color scale.
breaks_s_pa <- c(1e-04, 2e-04, 3e-04, 4e-04, 5e-04)
min_pa_s <- min(p_q_plot$pa.x, na.rm = TRUE)
max_pa_s <- max(p_q_plot$pa.x, na.rm = TRUE)

labels_pa_s <- c()
brks_pa_s   <- c(min_pa_s, breaks_s_pa, max_pa_s)
for (idx in 1:length(brks_pa_s)) {
  labels_pa_s <- c(labels_pa_s, brks_pa_s[idx + 1])
}
labels_pa_s <- labels_pa_s[1:length(labels_pa_s) - 1]

p_q_plot$brks <- cut(p_q_plot$pa.x,
                     breaks = brks_pa_s,
                     include.lowest = TRUE,
                     labels = labels_pa_s)

brks_pa_s_scale   <- as.numeric(levels(p_q_plot$brks))
labels_pa_s_scale <- rev(brks_pa_s_scale)

brks_pa_s_scale <- brks_pa_s_scale[1:5]

# Use ggplot2 to visualize the relationship between Q and the implied peak month,
# while coloring points by estimated P and overlaying theoretical lines for fixed P values.
library(ggplot2)

png("Tpq_qa_col_p.png", width = 750, height = 875)

ggplot() +
  labs(x = "Q adoption") +
  labs(y = "Peak month") +
  xlim(0.05, 0.3) + ylim(45, 95) +

  geom_point(data = p_q_plot,
             aes(x = qa.x, y = Tpq, colour = pa.x), size = I(4)) +
  theme_bw() +

  scale_colour_gradientn("P adoption", colours = rainbow(4),
                         breaks = as.numeric(brks_pa_s_scale),
                         labels = c(expression("1 x" ~ 10^-4), expression("2 x" ~ 10^-4),
                                    expression("3 x" ~ 10^-4), expression("4 x" ~ 10^-4),
                                    expression("5 x" ~ 10^-4))) +

  guides(colour = guide_colourbar(title.position = "top", key.position = "top",
                                  title.hjust = 0.5, labels = FALSE,
                                  barwidth = 33, label.hjust = 1)) +

  geom_line(data = data_OSN, aes(x = qa, y = T1), colour = "red") +
  geom_line(data = data_OSN, aes(x = qa, y = T2), colour = "orange") +
  geom_line(data = data_OSN, aes(x = qa, y = T3), colour = "#1affcc") +
  geom_line(data = data_OSN, aes(x = qa, y = T4), colour = "blue") +

  theme(panel.grid.major = element_line(color = "white", size = 0.2),
        panel.grid.minor = element_blank(),
        plot.background  = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        legend.background = element_rect(fill = "white", color = NA)) +

  theme(legend.box = "horizontal", legend.position = "bottom",
        legend.text = element_text(size = 20, hjust = 0, color = "#4e4d47"),
        legend.title = element_text(size = 23)) +

  theme(axis.text = element_text(size = 20),
        axis.title = element_text(size = 40),
        axis.text.x = element_text(vjust = -5),
        axis.title.x = element_text(vjust = -5)) +

  theme(plot.margin = unit(c(0.8, 1.2, 1.8, 1.2), "cm")) +

  annotate("text", x = 0.27, y = 95, colour = "black", label = "Fixed P", size = 8) +

  annotate("segment", x = 0.22, xend = 0.25, y = 90, yend = 90, colour = "red",     lwd = 2) +
  annotate("segment", x = 0.22, xend = 0.25, y = 87, yend = 87, colour = "orange",  lwd = 2) +
  annotate("segment", x = 0.22, xend = 0.25, y = 84, yend = 84, colour = "#1affcc", lwd = 2) +
  annotate("segment", x = 0.22, xend = 0.25, y = 81, yend = 81, colour = "blue",    lwd = 2) +

  annotate("text", x = 0.28, y = 90, colour = "black", size = 7, label = "5e-5") +
  annotate("text", x = 0.28, y = 87, colour = "black", size = 7, label = "1e-4") +
  annotate("text", x = 0.28, y = 84, colour = "black", size = 7, label = "3e-4") +
  annotate("text", x = 0.28, y = 81, colour = "black", size = 7, label = "5e-4") +

  scale_x_continuous(position = "top", limits = c(0.05, 0.3))

dev.off()



##########################################
# 4. Toy example of the BASS ABM
##########################################

# This final section simulates diffusion directly on a social network,
# illustrating how a Bass-like process can be implemented as an agent-based model (ABM)
# on a small sample network of 1,000 nodes.

########################
# 4.1 Read Data
########################

# Read network nodes (vertices) and links (edges).
ver <- read.csv("vertices_sample.csv")
edg <- read.csv("edges_sample.csv")

########################
# 4.2 Create the graph
########################

# Convert the edge list and vertex list into an undirected igraph object.
# NOTE: The original script assumes that igraph is installed and loaded.
network_fs <- graph_from_data_frame(edg, directed = FALSE, vertices = ver)

########################
# 4.3 Initialize nodes and parameters
########################

# Get the graph vertices.
nodes_fs <- V(network_fs)

# Initialize adopters:
# all vertices with month == 3 are treated as initial diffusers.
diffusers_fs    <- which(ver$month == 3)
susceptibles_fs <- setdiff(nodes_fs, diffusers_fs)

# Create a vector storing the adoption time of each node.
# Initial adopters are assigned time 1.
time_fs <- numeric(length(nodes_fs))
time_fs[diffusers_fs] <- 1

# Bass-style innovation and imitation parameters used in the simulation.
p_fs <- 0.000104
q_fs <- 0.12

########################
# 4.4 Precompute neighborhood information
########################

# Precompute each node's first-order ego network (its immediate neighborhood).
# This reduces repeated neighbor lookup during the simulation.
tic("Precompute neighborhoods")
neigh <- ego(network_fs, 1, nodes = nodes_fs)
toc()

########################
# 4.5 Simulate diffusion
########################

# Set random seed for reproducibility.
set.seed(1042)

# Start the diffusion clock after the initial adopters.
time_adoption <- 2

tic("Diffusion process")
while (time_adoption < 129) {
  tic(paste("Time step", time_adoption))

  # For each node, calculate the fraction of its neighbors who have already adopted.
  # length(x) - 1 subtracts the ego node itself from the ego network.
  a <- vapply(neigh,
              function(x) sum(x %in% diffusers_fs) / (length(x) - 1),
              numeric(1))

  # Bass-like adoption probability:
  #   p_fs = spontaneous innovation probability
  #   a * q_fs = imitation pressure from adopted neighbors
  prob <- p_fs + a * q_fs

  # Susceptible nodes adopt if a random draw is smaller than their adoption probability.
  adopters <- susceptibles_fs[runif(length(susceptibles_fs)) < prob[susceptibles_fs]]

  # Update adoption time and the sets of diffusers / susceptibles.
  time_fs[adopters] <- time_adoption
  diffusers_fs      <- c(diffusers_fs, adopters)
  susceptibles_fs   <- setdiff(nodes_fs, diffusers_fs)

  # Drop already adopted nodes from the neighborhood list to save computation.
  neigh <- neigh[setdiff(nodes_fs, diffusers_fs)]

  # Print cumulative number of adopters and current time step.
  cat(sum(time_fs != 0), "-->", time_adoption, "\n")

  time_adoption <- time_adoption + 1
  toc()
}
toc()

########################
# 4.6 Plot Results
########################

# Build a simple cumulative diffusion curve:
#   Time = adoption month
#   CDF  = cumulative fraction of adopters in the network
results <- data.frame(
  Time = 1:max(time_fs),
  CDF  = cumsum(tabulate(time_fs)) / vcount(network_fs)
)

# Plot the cumulative adoption trajectory.
plot(results$Time, results$CDF, type = "b",
     ylab = "CDF", xlab = "Time")



#########################
# Full Network ABM
#########################

# ABM simulation at https://github.com/bokae/spatial_diffusion
