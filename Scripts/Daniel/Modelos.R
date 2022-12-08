###Limpiar 
rm(list = ls())
cat("\014")

#Importar paquetes y cargar librerías
require(pacman)
#install.packages("xgboost")

p_load(tidyverse, rvest, data.table, dplyr, skimr, caret, rio, 
       vtable, stargazer, ggplot2, boot, MLmetrics, lfe,
       tidyverse, fabricatr, stargazer, Hmisc, writexl, viridis, here,
       modelsummary, # tidy, msummary
       gamlr,        # cv.gamlr
       ROCR, # ROC curve
       class, readxl, writexl,
       glmnet, janitor, doParallel, 
       rattle, fastDummies, tidymodels, themis, AER, randomForest,xgboost, ranger)


#directorio Daniel  
setwd("C:/Users/danie/OneDrive/Escritorio/Uniandes/PEG/Big Data and Machine Learning/BD-ML---FinalProject/Data_definitiva")

Base <- read.csv("Base_Completa.csv")

#Crear la variable y
Base <- Base%>%mutate(Var_y = valor_total/(12*I_HOGAR), log_y = log(Var_y))

Base_DEF <- Base%>%drop_na()%>%subset(I_HOGAR > 0)
View(Base_DEF%>%select(valor_total, I_HOGAR, Var_y))

hist(Base_DEF$log_y, breaks = 100)

#Revisar que no haya quedado ningún NA
sapply(Base_DEF, function(y) sum(length(which(is.na(y)))))
sum(is.na(Base_DEF))

#Generar las variables que son factores
names(Base_DEF)

Variables_categoricas <- c("Max_educ", "alguien_estudia", "transporte", 
                           "tiempo_moda_estudio", "tiempo_max_estudio", "tiempo_min_estudio",
                           "beca", "subsidio", "credito", "arte", "ciencia", "deportes",
                           "grupos_estudio", "parque", "lectura", "juegos", "alguna_extra",
                           "moda_estado", "madres_jovenes", "P5230", "P9090", "P784S1",
                           "P1077S1", "P1077S2", "P1077S3", "P1077S4", "P1077S5",
                           "P1077S6", "P1077S7", "P1077S8", "P1077S9", "P1077S10",
                           "P1077S14", "P1077S15", "P1077S16", "P1077S17", "P1077S19",
                           "P1077S21", "P1077S22", "P1077S23", "P1075", "P1913S2",
                           "P3353", "P3354", "P5095", "CLASE", "P2102", "P1070",
                           "P4005", "P4015", "P4567", "P8520S1A1", "P8520S5", "P8520S3",
                           "P8520S4") 

for (variable in Variables_categoricas) {
  a <- variable
  Base_DEF[,a] <- as.factor(Base_DEF[,a])
}

class(Base_DEF$P8520S1A1)

#Quitar las variables que sobran

Base_final <- Base_DEF%>%select(-tot_mayores_10, -tot_mayores_15, -tot_personas,
                                -Trabajo._15_años, -Trabajo._15_años, -Desempleo._15_años,
                                -Desempleo._totales, -Oficios._15_años, -Oficios._totales,
                                -Estudiante._15_años, -Estudiante._totales, -Incapacidad._15_años,
                                -Incapacidad._totales, -Otra._15_años, -Otra._totales,
                                -cotizan_pension, -cotizan_pension_tot, -cotizan_pension_15,
                                -SECUENCIA_P.y, -valor_matricula, -valor_uniformes, -valor_utiles,
                                -valor_pension, -valor_transporte, -valor_alimento, -valor_total, -alguien_estudia)

#Se guarda la base final definitiva
write.csv(Base_final, "La_base_definitiva_final.csv", row.names = FALSE)


#Esto de abajo realmente no se está usando
Base_modelo <- Base_final%>%select(-DIRECTORIO,-SECUENCIA_P.x,-Var_y,-alguna_extra,-ayudas_total)
matriz <- model.matrix(~ tiempo_promedio_transporte + alguien_estudia,Base_modelo)%>%data.frame()
class(Base_modelo$alguien_estudia)
names(Base_modelo)
Base_modelo$Max_educ%>%count()



#Se carga la base final
Base_final <- read.csv("La_base_definitiva_final.csv")
Base_final <- Base_final%>%subset(Var_y <=1)

#Se crea el train y el test
#Armar un test
#Escoger 200 datos aleatoriamente
#Se establece semilla
set.seed(1000)
n <- 0.8*nrow(Base_final)
smp_size <- floor(n)
train_ind <- sample(1:n, size = smp_size)
#Crear train set para ajustar los parámetros
train_2 <- Base_final[train_ind, ]
#Crear test set para evaluar el modelo
test <- Base_final[-train_ind, ]
#crear el de validación
#Se establece semilla
set.seed(1000)
n <- 0.8*nrow(train_2)
smp_size <- floor(n)
train_ind <- sample(1:n, size = smp_size)
#Crear train set para ajustar los parámetros
train_def <- train_2[train_ind, ]
#Crear test set para evaluar el modelo
valid <- train_2[-train_ind, ]


#####Ojo con las actividades extracurriculares
lambda <- 10^seq(-2, 3, length = 100)

#Modelo Lasso
lasso <- train(
  log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, data = train_def, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda=lambda), preProcess = c("center", "scale"))

lasso


lasso <- train(
  log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, data = train_def, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda=0.1), preProcess = c("center", "scale"))

lasso


#Modelo Ridge
ridge <- train(
  log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, data = train_def, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda=lambda), preProcess = c("center", "scale"))

ridge

ridge <- train(
  log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, data = train_def, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda=0.02535364), preProcess = c("center", "scale"))

ridge

#Modelo Elastic Net
EN <- train(
  log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, data = train_def, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = seq(0,1,0.05), lambda= lambda), preProcess = c("center", "scale"))

EN


EN <- train(
  log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, data = train_def, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0.95, lambda= 0.01), preProcess = c("center", "scale"))

EN

#Sigue un RF

sqrt(73) #8.5
#hiper grillita
hyper_grid <- expand.grid(
  mtry       = c(8,9),
  max_depht  = c(5,8,10),
  sampe_size = c(0.6,1),
  num_trees = c(500, 1000, 1500),
  OOB_RMSE   = 0)

#Se crea la base que guarda los resultados
resultadoRF <- data.frame()

#Corremos el for para cada combinación de la hipergrilla
for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, 
    data            = train_def, 
    num.trees       = hyper_grid$num_trees[i],
    mtry            = hyper_grid$mtry[i],
    max.depth       = hyper_grid$max_depht[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123, # Notese el seteo de la semilla
  )
  
  pred <- predict(model, valid)
  pred <- as.data.frame(pred)
  pred$real <- valid$log_y
  colnames(pred) <- c("pred","real")
  pred$error <- exp(pred$pred) - exp(pred$real)
  pred$error_funcion <- ifelse(pred$error >0, pred$error^2, abs(pred$error))
  avg = mean(pred$error_funcion)
  result <- data.frame(Modelo = "RF_ranger",
                       AVG = avg,
                       numtrees = hyper_grid$num_trees[i],
                       mtryRF = hyper_grid$mtry[i],
                       maxdepth = hyper_grid$max_depht[i],
                       samplefraction = hyper_grid$sampe_size[i])
  resultadoRF <- bind_rows(resultadoRF, result)
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

#Revisar resultados
resultadoRF
which.min(resultadoRF$AVG) #12
hyper_grid
which.min(hyper_grid$OOB_RMSE) #36


#entrenar el modelo con los parámetros del mejor modelo para minimizar OOBRMSE
model <- ranger(
  formula         = log_y ~ . -DIRECTORIO-SECUENCIA_P.x-Var_y-alguna_extra-ayudas_total, 
  data            = train_def, 
  num.trees       = 500,
  mtry            = 9,
  max.depth       = 10,
  sample.fraction = 1,
  seed            = 123, # Notese el seteo de la semilla
  #importance      = "impurity" 
)

#predicción en el test
pred <- predict(model, test)

hist(pred$predictions)
hist(test$log_y)

RMSE(pred$predictions, test$log_y)
#Un XGboost

#Si es posible un superlearner

#PCA

#En el XGboost podría ser interesante ver una función de pérdida distinta (con eso salen ya al menos 6 modelos)



#Para predecir el R2
R2(Base_final$log_y, Base_final$predicciones)

