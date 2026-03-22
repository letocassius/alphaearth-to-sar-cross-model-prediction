# AlphaEarth Embedding Diagnostics

library(tidyverse)
library(corrplot)
library(car)
library(GGally)
library(broom)

# 1. Load data

embeddings <- read.csv("alphaearth-to-sar-cross-model-prediction/DataSources/AlphaEarth_Embeddings_2024.csv")

# remove non-numeric columns
embeddings_clean <- embeddings %>%
  select(-`system.index`, -contains("geo"), everything()) %>%
  select(where(is.numeric))

# 2. Summary statistics for each embedding dimension

summary_stats <- embeddings_clean %>%
  pivot_longer(cols = everything(),
               names_to = "embedding",
               values_to = "value") %>%
  group_by(embedding) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    min = min(value, na.rm = TRUE),
    q25 = quantile(value, .25, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    q75 = quantile(value, .75, na.rm = TRUE),
    max = max(value, na.rm = TRUE),
    skewness = mean((value - mean(value))^3) / sd(value)^3
  )

print(summary_stats)

# 3. Distribution visualization

embeddings_clean %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 40, fill = "steelblue") +
  facet_wrap(~name, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of AlphaEarth Embedding Dimensions")

# 4. Correlation matrix

cor_matrix <- cor(embeddings_clean, use = "pairwise.complete.obs")

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.cex = .7,
  tl.col = "black"
)

# 5. Identify highly correlated features

high_corr <- as.data.frame(as.table(cor_matrix)) %>%
  filter(
    Var1 != Var2,
    abs(Freq) > 0.9
  )

print(high_corr)
summary(embeddings)

# 6. Variance Inflation Factor

# Create dummy response variable for VIF calculation
set.seed(1)
y <- rnorm(nrow(embeddings_clean))

model <- lm(y ~ ., data = embeddings_clean)

vif_values <- vif(model)

vif_table <- tibble(
  embedding = names(vif_values),
  VIF = vif_values
) %>%
  arrange(desc(VIF))

print(vif_table, n=65)

# 7. Principal Component Analysis

pca <- prcomp(embeddings_clean, scale. = TRUE)

pca_variance <- tibble(
  PC = 1:length(pca$sdev),
  variance_explained = (pca$sdev^2) / sum(pca$sdev^2)
)

ggplot(pca_variance, aes(PC, variance_explained)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "PCA Scree Plot",
    y = "Variance Explained"
  )

# cumulative variance
pca_variance %>%
  mutate(cumulative = cumsum(variance_explained)) %>%
  print(n=65)

# 8. Pairwise embedding relationships

embeddings_clean %>%
  select(1:6) %>% 
  ggpairs()



## Backward selection: AIC

library(MASS)

backward_aic <- stepAIC(
  model,
  direction = "backward",
  trace = TRUE
)

summary(backward_aic)

selected_aic <- names(coef(backward_aic))[-1]
selected_aic

# Run backward AIC
library(MASS)

backward_aic <- stepAIC(
  model,
  direction = "backward",
  trace = TRUE
)

# Extract AIC path
aic_path <- backward_aic$anova

aic_path

library(ggplot2)

aic_path$step <- 1:nrow(aic_path)

ggplot(aic_path, aes(x = step, y = AIC)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(
    title = "Backward Selection AIC Path",
    x = "Step",
    y = "AIC"
  )

#BIC 

n <- nrow(embeddings_clean)

backward_bic <- step(
  model,
  direction = "backward",
  k = log(n),   
  trace = TRUE
)

summary(backward_bic)

selected_bic <- names(coef(backward_bic))[-1]
selected_bic

length(selected_aic)
length(selected_bic)

tibble(
  method = c("AIC", "BIC"),
  variables = list(selected_aic, selected_bic)
) 

# BIC is heavily penalizing the model

# Using leaps

library(tidyverse)
library(leaps)

# same y used for VIF model
set.seed(1)
y <- rnorm(nrow(embeddings_clean))

# combine into one modeling data frame
subset_data <- embeddings_clean %>%
  mutate(y = y)

# best subset selection
best_subsets <- regsubsets(
  y ~ .,
  data = subset_data,
  nvmax = ncol(embeddings_clean),   # try all subset sizes
  method = "exhaustive"
)

best_summary <- summary(best_subsets)

