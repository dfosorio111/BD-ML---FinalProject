### MODELOS DIEGO

#Limpiar el ambiente
rm(list=ls())
#Establecer directorios
#Diego
setwd("C:/Users/Diego/OneDrive/Documents/GitHub/BD-ML---FinalProject/Scripts/Diego")



#Importar paquetes y cargar librerÃ­as
require(pacman)
p_load(tidyverse, rvest, data.table, dplyr, skimr, caret, rio, 
       vtable, stargazer, ggplot2, boot, MLmetrics, lfe, 
       tidyverse, fabricatr, stargazer, Hmisc, writexl, viridis, here,
       modelsummary, # tidy, msummary
       gamlr,        # cv.gamlr
       ROCR, # ROC curve
       class, glmnet, janitor, doParallel, rattle, fastDummies, tidymodels, themis, AER)


# cargar base 

data <- read.csv("data/La_base_definitiva_final.csv")
data <- data%>%subset(Var_y<=1)

# Check los NAs de la base
sapply(data, function(x) sum(is.na(x)))
sum(is.na(data))

#Revisemos quÃ© variables tenemos
names(data)

# evaluar correlacion entre variables
cor(data$transporte, data$log_y)

# generar variables categoricas
Variables_categoricas <- c("Max_educ", "transporte", 
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

#vars_num <- data[, (!names(data) %in% Variables_categoricas)]

for (variable in Variables_categoricas) {
  a <- variable
  data[,a] <- as.factor(data[,a])
}

# crear matriz de dummies con las variables categoricas
df <- model.matrix(~. -DIRECTORIO-SECUENCIA_P.x-1, data) %>% data.frame()


# Dividimos train/test (80/20)
# split los datos de entrenamiento en train-set y test-set


# crear 3 bases: train, validation, test
# split los datos de entrenamiento en train-set y test-set  80-20%
set.seed(100)
n <- nrow(df)
smp_size <- floor(0.8*n)
train_obs <- sample(floor(nrow(df)*0.8))

# crear X_train para ajustar/entrenar los modelos
X_train <- df[train_obs, ]

# split X_train en train-set y validation-set  70/30%
set.seed(100)
val_obs <- sample(floor(nrow(X_train)*0.3))

# crear X_train y X_val
X_val <- X_train[val_obs,]
X_train <- X_train[-val_obs,]

# crear X_test
X_test <- df[-train_obs, ]

# crear y 
y_train <- X_train[,"log_y"]
y_val <- X_val[,"log_y"]
y_test <- X_test[,"log_y"]

# quitar y de las bases de X (variables)
drop_logy <- c("log_y")

X_train <- X_train[, (!names(X_train) %in% drop_logy)]
X_val <- X_val[, (!names(X_val) %in% drop_logy)]
X_test <- X_test[, (!names(X_test) %in% drop_logy)]

# guardar variable de y gasto educacion real
price_train <- X_train$Var_y
price_val <- X_val$Var_y
price_test <- X_test$Var_y

# quitar price de las bases de X (variables)
drop_Var_y <- c("Var_y")

X_train <- X_train[, (!names(X_train) %in% drop_Var_y)]
X_val <- X_val[, (!names(X_val) %in% drop_Var_y)]
X_test <- X_test[, (!names(X_test) %in% drop_Var_y)]


# lista de variables numericas
variables_numericas <- c("tiempo_promedio_transporte", "monto_beca", "monto_subsidio",            
                         "monto_credito", "ayudas_total", "P5000", "P5010", "CANT_PERSONAS_HOGAR",       
                         "I_HOGAR", "I_UGASTO", "PERCAPITA", "Trabajo._10_años", "Trabajo._totales",          
                         "Desempleo._10_años", "Oficios._10_años", "Estudiante._10_años", "Incapacidad._10_años", "Otra._10_años",            
                         "cotizan_pension_10", "valor_arriendos", "CANT_HOGARES_VIVIENDA")  



# Estandarizar VARIABLES NUMERICAS (CONTINUAS) DESPUES de hacer split de la base de datos 
# crear escalador
escalador <- preProcess(X_train[, variables_numericas])

train_s <- X_train
val_s <- X_val
test_s <- X_test

train_s[, variables_numericas] <- predict(escalador, X_train[, variables_numericas])
val_s[, variables_numericas] <- predict(escalador, X_val[, variables_numericas])
test_s[, variables_numericas] <- predict(escalador, X_test[, variables_numericas])

# convertir bases a dataframe
train_s <- data.frame(train_s)
val_s <- data.frame(val_s)
test_s <- data.frame(test_s)

X_train <- data.frame(X_train)
X_val <- data.frame(X_val)
X_test <- data.frame(X_test)


#  unificar bases completas
train_modelos <- data.frame(y_train, train_s)
train_modelos$y_train <- as.numeric(train_modelos$y_train)

val_modelos <- data.frame(y_val, val_s)
val_modelos$y_val <- as.numeric(val_modelos$y_val)

test_modelos <- data.frame(y_test, test_s)
test_modelos$y_test <- as.numeric(test_modelos$y_test)

### GUARDAR BASES PROCESADAS PARA REALIZAR MODELOS
write.csv(train_modelos, 'data/train_modelos.csv')
write.csv(val_modelos, 'data/val_modelos.csv')
write.csv(test_modelos, 'data/test_modelos.csv')


### MODELOS
# ELASTIC NET: 

# crear K-Fold particiones sobre la base para ajustar los modelos
set.seed(100)
cv5 <- trainControl(number = 5,
                    method = "cv")

# crear grid_search para elastic net
tunegrid_glmnet <- data.frame(alpha= seq(0,1,length.out=10),
                              lambda = seq(0,5,length.out=50) )

# crear modelo de Elastic Net
# hiper parametros: method: glmnet, preProcess centrar y escalar
# trainControl: CV-5Fold, metric optimizar: RMSE, tuneGri

modelo_glmnet <- train(y_train~. -alguna_extra1- ayudas_total,
                       data = train_modelos,
                       method = "glmnet",
                       trControl= cv5,
                       metric = 'RMSE',
                       tuneGrid = tunegrid_glmnet)
# preProcess = c("center", "scale"),
modelo_glmnet

# plot del modelo 
plot(modelo_glmnet)
glmnet_opt <- modelo_glmnet$bestTune


# Elastic Net  modelo óptimo
# enrtenar/ajustar modelo óptimo
modelo_en <- glmnet(
  x  = train_s,
  y = y_train,
  alpha = glmnet_opt$alpha,
  lambda = glmnet_opt$lambda,
  standardize = FALSE
  
)

# glmnet_opt$alpha = 0
# glmnet_opt$lambda = 0

summary(modelo_en)

# calcular vector de predicciones en datos de entrenamiento, validacion y prueba
y_predict_train <- predict(modelo_en, newx = as.matrix( train_s))
y_predict_val <- predict(modelo_en, newx = as.matrix( val_s))
y_predict_test <- predict(modelo_en, newx = as.matrix( test_s))


#Resultados:
res_modelo_en <-  as.data.frame(cbind(price_test,mean(price_test)) )

# crear variable de prediccion de precio (exp(valor prediccion)) en datos de validacion 
res_modelo_en <- res_modelo_en%>%mutate(y_predict_test = exp(y_predict_test))

# crear variable real de precio de educacion en datos de validacion
res_modelo_en <- res_modelo_en%>%mutate(y_real = price_test)


#colnames(res_modelo_en) <- c("Real", "Promedio", "Prediccion")
res_modelo_en$error <- res_modelo_en$y_predict_test - res_modelo_en$y_real
res_modelo_en$fun_error <- ifelse(res_modelo_en$y_predict_test>res_modelo_en$y_real, res_modelo_en$error^2, abs(res_modelo_en$error) )


rmse <- sqrt( mean(res_modelo_en$error^2))
mae <- mean(abs(res_modelo_en$error^2))
avg <- mean(res_modelo_en$fun_error)

metricas_en <- data.frame(Modelo = "Elastic Net",
                          AVG = avg,
                          RMSE = rmse,
                          MAE = mae
)

# GUARDAR MODELO DE EN
write_rds( modelo_en,'data/modelo_en.RDS')
# CARGAR MODELO DE LIN REG
modelo_en <- readRDS('data/modelo_en.RDS')

# GUARDAR DATAFRAME DE RESULTADOS LIN REG
write_rds(res_modelo_en,'data/resultados_en.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
res_modelo_en <- readRDS('data/resultados_en.RDS')

# GUARDAR METRICAS DE EVALUACION EN VALIDATION-SET
write_rds(metricas_en,'data/metricas_en.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
metricas_en <- readRDS('data/metricas_en.RDS')



# LASSO: ALPHA = 1, PENALTY = L1

# crear K-Fold particiones sobre la base para ajustar los modelos
set.seed(100)
cv5 <- trainControl(number = 5,
                    method = "cv")

# crear grid_search para elastic net
tunegrid_lasso <- data.frame(alpha= 1,
                             lambda = seq(0,5,length.out=50) )

# crear modelo de Elastic Net
# hiper parametros: method: glmnet, preProcess centrar y escalar
# trainControl: CV-5Fold, metric optimizar: RMSE, tuneGri

modelo_lasso <- train(y_train~. -alguna_extra1- ayudas_total,
                      data = train_modelos,
                      method = "glmnet",
                      trControl= cv5,
                      metric = 'RMSE',
                      tuneGrid = tunegrid_lasso)
# preProcess = c("center", "scale"),
modelo_lasso

# plot del modelo 
plot(modelo_lasso)
lasso_opt <- modelo_lasso$bestTune

# Lasso  modelo óptimo

# enrtenar/ajustar modelo óptimo
lasso <- glmnet(
  x  = train_s,
  y = y_train,
  alpha = lasso_opt$alpha,
  lambda = lasso_opt$lambda,
  standardize = FALSE
)

# lasso_opt$alpha = 1
# lasso_opt$lambda = 0.1020408


summary(lasso)

# calcular vector de predicciones en datos de entrenamiento, validacion y prueba
y_predict_train <- predict(lasso, newx = as.matrix( train_s))
y_predict_val <- predict(lasso, newx = as.matrix( val_s))
y_predict_test <- predict(lasso, newx = as.matrix( test_s))


#Resultados:
res_modelo_lasso <-  as.data.frame(cbind(price_test,mean(price_test)) )

# crear variable de prediccion de precio (exp(valor prediccion)) en datos de validacion 
res_modelo_lasso <- res_modelo_lasso%>%mutate(y_predict_test = exp(y_predict_test))

# crear variable real de precio de educacion en datos de validacion
res_modelo_lasso <- res_modelo_lasso%>%mutate(y_real = price_test)


#colnames(res_modelo_en) <- c("Real", "Promedio", "Prediccion")
res_modelo_lasso$error <- res_modelo_lasso$y_predict_test - res_modelo_lasso$y_real
res_modelo_lasso$fun_error <- ifelse(res_modelo_lasso$y_predict_test>res_modelo_lasso$y_real, res_modelo_lasso$error^2, abs(res_modelo_lasso$error) )


rmse <- sqrt( mean(res_modelo_lasso$error^2))
mae <- mean(abs(res_modelo_lasso$error^2))
avg <- mean(res_modelo_lasso$fun_error)

metricas_lasso <- data.frame(Modelo = "Lasso",
                             AVG = avg,
                             RMSE = rmse,
                             MAE = mae
)

# GUARDAR MODELO DE LIN REG
write_rds(lasso,'data/lasso.RDS')
# CARGAR MODELO DE LIN REG
lasso <- readRDS('data/lasso.RDS')

# GUARDAR DATAFRAME DE RESULTADOS LIN REG
write_rds(res_modelo_lasso,'data/res_modelo_lasso.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
res_modelo_lasso <- readRDS('data/res_modelo_lasso.RDS')

# GUARDAR METRICAS DE EVALUACION EN VALIDATION-SET
write_rds(metricas_lasso,'data/metricas_lasso.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
metricas_lasso <- readRDS('data/metricas_lasso.RDS')


# RIDGE: ALPHA = 0, PENALTY = L2

# crear K-Fold particiones sobre la base para ajustar los modelos
set.seed(100)
cv5 <- trainControl(number = 5,
                    method = "cv")

# crear grid_search para elastic net
tunegrid_ridge <- data.frame(alpha= 0,
                             lambda = seq(0,5,length.out=50) )

# crear modelo de Elastic Net
# hiper parametros: method: glmnet, preProcess centrar y escalar
# trainControl: CV-5Fold, metric optimizar: RMSE, tuneGri

modelo_ridge <- train(y_train~. -alguna_extra1- ayudas_total,
                      data = train_modelos,
                      method = "glmnet",
                      trControl= cv5,
                      metric = 'RMSE',
                      tuneGrid = tunegrid_ridge)
# preProcess = c("center", "scale"),
modelo_ridge

# plot del modelo 
plot(modelo_ridge)
ridge_opt <- modelo_ridge$bestTune


# Elastic Net  modelo óptimo

# enrtenar/ajustar modelo óptimo
ridge <- glmnet(
  x  = train_s,
  y = y_train,
  alpha = ridge_opt$alpha,
  lambda = ridge_opt$lambda,
  standardize = FALSE
)

summary(ridge)

# ridge_opt$alpha = 0
# ridge_opt$lambda = 1.530612

# calcular vector de predicciones en datos de entrenamiento, validacion y prueba
y_predict_train <- predict(ridge, newx = as.matrix( train_s))
y_predict_val <- predict(ridge, newx = as.matrix( val_s))
y_predict_test <- predict(ridge, newx = as.matrix( test_s))


#Resultados:
res_modelo_ridge <-  as.data.frame(cbind(price_test,mean(price_test)) )

# crear variable de prediccion de precio (exp(valor prediccion)) en datos de validacion 
res_modelo_ridge <- res_modelo_ridge%>%mutate(y_predict_test = exp(y_predict_test))

# crear variable real de precio de educacion en datos de validacion
res_modelo_ridge <- res_modelo_ridge%>%mutate(y_real = price_test)

#colnames(res_modelo_en) <- c("Real", "Promedio", "Prediccion")
res_modelo_ridge$error <- res_modelo_ridge$y_predict_test- res_modelo_ridge$y_real
res_modelo_ridge$fun_error <- ifelse(res_modelo_ridge$y_predict_test>res_modelo_ridge$y_real, res_modelo_ridge$error^2, abs(res_modelo_ridge$error))


rmse <- sqrt( mean(res_modelo_ridge$error^2))
mae <- mean(abs(res_modelo_ridge$error^2))
avg <- mean(res_modelo_ridge$fun_error)

metricas_ridge <- data.frame(Modelo = "Ridge",
                             AVG = avg,
                             RMSE = rmse,
                             MAE = mae
)

# GUARDAR MODELO DE LIN REG
write_rds(ridge,'data/ridge.RDS')
# CARGAR MODELO DE LIN REG
ridge <- readRDS('data/ridge.RDS')

# GUARDAR DATAFRAME DE RESULTADOS LIN REG
write_rds(res_modelo_ridge,'data/res_modelo_ridge.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
res_modelo_ridge <- readRDS('data/res_modelo_ridge.RDS')

# GUARDAR METRICAS DE EVALUACION EN VALIDATION-SET
write_rds(metricas_ridge,'data/metricas_ridge.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
metricas_ridge <- readRDS('data/metricas_ridge.RDS')
