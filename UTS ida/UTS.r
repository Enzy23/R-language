# Memuat library yang diperlukan
library(readxl)
library(tidyverse)
library(cluster)
library(randomForest)
library(e1071)
library(caret)
library(glmnet)

# Baca file Excel
nama_file <- "./Mall_Customers.xlsx"
data_excel <- read_excel(nama_file)

# Mengubah nama kolom
data_excel <- data_excel %>%
  rename(
    customer_id = CustomerID,
    gender = Genre,
    age = Age,
    annual_income = `Annual Income (k$)`,
    spending_score = `Spending Score (1-100)`
  )

# Menampilkan ringkasan statistik
summary(data_excel)

# Visualisasi distribusi variabel numerik
# Histogram Usia
ggplot(data_excel, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(title = "Distribusi Usia", x = "Usia", y = "Frekuensi")

# Hubungan antara Pendapatan Tahunan dan Skor Pengeluaran
ggplot(data_excel, aes(x = annual_income, y = spending_score)) +
  geom_point(color = "purple") +
  labs(title = "Hubungan Antara Pendapatan Tahunan dan Skor Pengeluaran", x = "Pendapatan Tahunan", y = "Skor Pengeluaran")

# Visualisasi distribusi variabel kategorikal
# Barplot Gender
ggplot(data_excel, aes(x = gender, fill = gender)) +
  geom_bar() +
  labs(title = "Distribusi Gender", x = "Gender", y = "Frekuensi")

# Analisis lebih lanjut
# Density Plot Usia berdasarkan Gender
ggplot(data_excel, aes(x = age, fill = gender)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribusi Usia berdasarkan Gender", x = "Usia", y = "Density")

# Scatter plot Pendapatan Tahunan dan Skor Pengeluaran berdasarkan Gender
ggplot(data_excel, aes(x = annual_income, y = spending_score, color = gender)) +
  geom_point() +
  labs(title = "Hubungan Antara Pendapatan Tahunan dan Skor Pengeluaran (berdasarkan Gender)", x = "Pendapatan Tahunan", y = "Skor Pengeluaran")

# K-Means Clustering
# Pilih fitur yang akan digunakan untuk clustering
fitur_clustering <- select(data_excel, age, annual_income, spending_score)

# Normalisasi data
fitur_clustering_scaled <- scale(fitur_clustering)

# Menentukan jumlah cluster dengan metode elbow
wss <- (nrow(fitur_clustering_scaled) - 1) * sum(apply(fitur_clustering_scaled, 2, var))
for (i in 1:10) wss[i] <- sum(kmeans(fitur_clustering_scaled, centers = i)$withinss)
plot(1:10, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Berdasarkan plot, pilih jumlah cluster yang optimal, misalnya, 4
jumlah_cluster <- 4

# Melakukan clustering dengan K-Means
kmeans_model <- kmeans(fitur_clustering_scaled, centers = jumlah_cluster)
data_excel$cluster <- kmeans_model$cluster

# Pemisahan data
set.seed(123)  # Untuk reproduksi hasil
indeks_data_train <- sample(1:nrow(data_excel), 0.8 * nrow(data_excel))
data_train <- data_excel[indeks_data_train, ]
data_test <- data_excel[-indeks_data_train, ]

# Membuat fungsi untuk evaluasi model dan plotting
evaluasi_dan_plot <- function(model, nama_model, data_test) {
  prediksi_spending <- predict(model, newdata = data_test)
  mse <- mean((prediksi_spending - data_test$spending_score)^2)
  mae <- mean(abs(prediksi_spending - data_test$spending_score))
  rmse <- sqrt(mse)
  
  cat(nama_model, "Mean Squared Error (MSE):", mse, "\n")
  cat(nama_model, "Mean Absolute Error (MAE):", mae, "\n")
  cat(nama_model, "Root Mean Squared Error (RMSE):", rmse, "\n")
  
  # Plot hasil prediksi
  plot(data_test$annual_income, data_test$spending_score, main = nama_model, 
       xlab = "Annual Income", ylab = "Spending Score", pch = 16, col = "blue")
  points(data_test$annual_income, prediksi_spending, pch = 16, col = "red")
  legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), pch = 16)
}

# Regresi Linier
regresi_linier_model <- lm(spending_score ~ annual_income, data = data_train)
evaluasi_dan_plot(regresi_linier_model, "Regresi Linier", data_test)

# Regresi Polinomial
regresi_polinomial_model <- lm(spending_score ~ poly(annual_income, degree = 2), data = data_train)
evaluasi_dan_plot(regresi_polinomial_model, "Regresi Polinomial", data_test)

# Regresi Random Forest
rf_model <- randomForest(spending_score ~ annual_income, data = data_train, ntree = 500)
evaluasi_dan_plot(rf_model, "Random Forest", data_test)

# Support Vector Machine (SVM)
svm_model <- svm(spending_score ~ annual_income, data = data_train)
evaluasi_dan_plot(svm_model, "Support Vector Machine", data_test)

# Bar plot untuk komparasi performa model
model_names <- c("Regresi Linier", "Regresi Polinomial", "Random Forest", "SVM")
mse_values <- c(
  mean((predict(regresi_linier_model, newdata = data_test) - data_test$spending_score)^2),
  mean((predict(regresi_polinomial_model, newdata = data_test) - data_test$spending_score)^2),
  mean((predict(rf_model, newdata = data_test) - data_test$spending_score)^2),
  mean((predict(svm_model, newdata = data_test) - data_test$spending_score)^2)
)
mae_values <- c(
  mean(abs(predict(regresi_linier_model, newdata = data_test) - data_test$spending_score)),
  mean(abs(predict(regresi_polinomial_model, newdata = data_test) - data_test$spending_score)),
  mean(abs(predict(rf_model, newdata = data_test) - data_test$spending_score)),
  mean(abs(predict(svm_model, newdata = data_test) - data_test$spending_score))
)
rmse_values <- c(
  sqrt(mean((predict(regresi_linier_model, newdata = data_test) - data_test$spending_score)^2)),
  sqrt(mean((predict(regresi_polinomial_model, newdata = data_test) - data_test$spending_score)^2)),
  sqrt(mean((predict(rf_model, newdata = data_test) - data_test$spending_score)^2)),
  sqrt(mean((predict(svm_model, newdata = data_test) - data_test$spending_score)^2))
)

performa_model <- data.frame(Model = model_names, MSE = mse_values, MAE = mae_values, RMSE = rmse_values)

# Bar plot untuk komparasi MSE
ggplot(performa_model, aes(x = Model, y = MSE, fill = Model, label = round(MSE, 2))) +
  geom_bar(stat = "identity") +
  geom_text(position = position_stack(vjust = 0.5), color = "white") +
  labs(title = "Komparasi MSE Model Regresi", x = "Model Regresi", y = "Mean Squared Error (MSE)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Bar plot untuk komparasi MAE
ggplot(performa_model, aes(x = Model, y = MAE, fill = Model, label = round(MAE, 2))) +
  geom_bar(stat = "identity") +
  geom_text(position = position_stack(vjust = 0.5), color = "white") +
  labs(title = "Komparasi MAE Model Regresi", x = "Model Regresi", y = "Mean Absolute Error (MAE)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Bar plot untuk komparasi RMSE
ggplot(performa_model, aes(x = Model, y = RMSE, fill = Model, label = round(RMSE, 2))) +
  geom_bar(stat = "identity") +
  geom_text(position = position_stack(vjust = 0.5), color = "white") +
  labs(title = "Komparasi RMSE Model Regresi", x = "Model Regresi", y = "Root Mean Squared Error (RMSE)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


