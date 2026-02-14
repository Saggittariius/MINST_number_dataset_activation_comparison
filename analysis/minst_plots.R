# ============================================
# MNIST Activation Function Comparison
# Analysis and Visualization Script
# ============================================

# Load required libraries
library(ggplot2)
library(gridExtra)

# ============================================
# 1. DATA LOADING
# ============================================

# Read individual CSV files
sigmoid_data <- read.csv("nminst_results_sigmoid.csv")
relu_data <- read.csv("nminst_results_relu.csv")
leaky_data <- read.csv("nminst_results_leaky_relu.csv")
softmax_data <- read.csv("nminst_results_softmax.csv")
torch_data <- read.csv("nminst_results_pytorch.csv")

# Combine into one dataset
all_data <- rbind(sigmoid_data, relu_data, leaky_data, softmax_data, torch_data)

# ============================================
# 2. DATA CLEANING
# ============================================

# Remove failed runs (accuracy < 15%)
all_data_clean <- subset(all_data, acc > 0.15)

# ============================================
# 3. DATA AGGREGATION
# ============================================

# Calculate average accuracy and loss per model per epoch
avg_data <- aggregate(cbind(acc, loss, time) ~ model + epoch, 
                      data = all_data_clean, 
                      FUN = mean)

# Create versions without entropy loss and without MSE
avg_data_no_CE <- subset(avg_data, !(model %in% c("softmax_relu", "pytorch_relu")))
avg_data_CE <- subset(avg_data, (model %in% c("softmax_relu", "pytorch_relu")))

# Get final epoch data for distribution plot
final_epoch <- subset(all_data_clean, epoch == 10)

# Calculate average time per epoch for each model
time_summary <- aggregate(time ~ model, data = avg_data, FUN = mean)

# ============================================
# 4. CREATE PLOTS
# ============================================

# Plot 1: Accuracy over epochs
time_text <- paste(time_summary$model, ":", round(time_summary$time, 2), "s", collapse = " | ")

p1 <- ggplot(avg_data, aes(x = epoch, y = acc, color = model)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = 0:10) +
  labs(title = "Accuracy Over Epochs",
       subtitle = paste("Time/epoch:", time_text),
       x = "Epoch",
       y = "Accuracy",
       color = "Model") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        plot.subtitle = element_text(hjust = 0.5, size = 7),
        legend.position = "bottom")

# Plot 2: Loss over epochs (MSE only)
p2 <- ggplot(avg_data_no_CE, aes(x = epoch, y = loss, color = model)) +
  geom_line(size = 0.8) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = 0:10) +
  labs(title = "Model Loss Comparison on MNIST (MSE)",
       x = "Epoch",
       y = "MSE Loss",
       color = "Activation Function") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        legend.position = "bottom")

# Plot 3: Final accuracy distribution
p3 <- ggplot(final_epoch, aes(x = model, y = acc, color = model)) +
  geom_jitter(width = 0.2, size = 3, alpha = 0.6) +
  stat_summary(fun = mean, geom = "point", size = 4, shape = 18, color = "black") +
  scale_color_brewer(palette = "Set1") +
  labs(title = "Final Accuracy Distribution",
       x = "Model",
       y = "Accuracy",
       color = "Model") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        legend.position = "bottom")

# Plot 4: Cross Entropy comparison
p4 <- ggplot(avg_data_CE, aes(x = epoch, y = loss, color = model)) +
  geom_line(size = 0.8) +
  geom_point(size = 2.5) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = 0:10) +
  labs(title = "Model Loss Comparison on MNIST (CE)",
  x = "Epoch",
  y = "Cross Entropy Loss",
  color = "Activation Function") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
  legend.position = "bottom",
  panel.background = element_rect(fill = "white", color = NA),
  plot.background = element_rect(fill = "white", color = NA)) 


# ============================================
# 5. DISPLAY AND SAVE
# ============================================

# Display combined plot
combined <- grid.arrange(p1, p2, p3, ncol = 3)

# Save to file
ggsave("mnist_activation_comparison.png", combined, 
       width = 15, height = 5, dpi = 300)

ggsave("mnist_cross_entropy_loss_comparison.png", p4, 
       width = 5, height = 5, dpi = 300)

print("Analysis complete! Plot saved as 'mnist_activation_comparison.png'")