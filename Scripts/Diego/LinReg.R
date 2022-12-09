### MODELOS DIEGO

#Limpiar el ambiente
rm(list=ls())
#Establecer directorios
#Diego
setwd("C:/Users/df.osorio11/Documents/aux_docs/bd_econ/PF")



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
                         "I_HOGAR", "I_UGASTO", "PERCAPITA", "Trabajo._10_aÃ.os", "Trabajo._totales",          
                         "Desempleo._10_aÃ.os", "Oficios._10_aÃ.os", "Estudiante._10_aÃ.os", "Incapacidad._10_aÃ.os", "Otra._10_aÃ.os",            
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
write_rds(train_modelos, 'data/train_modelos.csv')
write_rds(val_modelos, 'data/val_modelos.csv')
write_rds(test_modelos, 'data/test_modelos.csv')


### MODELOS

# REGRESION LINEAL

# ajustar/entrenar modelos con train-set
# quitar la variable price
lin_reg <- lm(y_train~. -alguna_extra1- ayudas_total, data = train_s)
summary(lin_reg)

### GUARDAR MODELO LIN REG. ENTRENADO
write_rds(lin_reg,'modelos/lin_reg.RDS')


# calcular vector de predicciones en datos de entrenamiento, validacion y prueba
y_predict_train <- predict(lin_reg, train_s)
y_predict_val <- predict(lin_reg, val_s)
y_predict_test <- predict(lin_reg, test_s)


# Predictores (coeficientes)
lin_reg_coeff <- lin_reg$coefficients %>%
  enframe(name = "predictor", value = "coeficiente")

lin_reg_coeff %>%
  filter(predictor != "`(Intercept)`") %>%
  ggplot(aes(x = reorder(predictor, abs(coeficiente)), 
             y = coeficiente)) +
  geom_col(fill = "darkblue") +
  coord_flip() +
  labs(title = "Coeficientes de Modelo Lin. Reg", 
       x = "Variables",
       y = "Coeficientes") +
  theme_bw()

#Resultados:
res_lin_reg <-  as.data.frame(cbind(price_val,mean(price_val)) )

# crear variable de prediccion de precio (exp(valor prediccion))
res_lin_reg <- res_lin_reg%>%mutate(y_predict_val = exp(y_predict_val))

# crear variable real de precio de educacion en datos de validacion
res_lin_reg <- res_lin_reg%>%mutate(y_real = price_val)


#colnames(res_lin_reg) <- c("Real", "Promedio", "Prediccion")
res_lin_reg$error <- res_lin_reg$y_real- res_lin_reg$y_predict_val
res_lin_reg$fun_error <- ifelse(res_lin_reg$y_predict_val>res_lin_reg$y_real, res_lin_reg$error^2, ifelse(res_lin_reg$y_predict_val<res_lin_reg$y_real,res_lin_reg$error,res_lin_reg$error))


rmse <- sqrt( mean(res_lin_reg$error^2))
mae <- mean(abs(res_lin_reg$error^2))
avg <- mean(res_lin_reg$fun_error)

metricas_lm <- data.frame(Modelo = "Regresión Lineal",
                          AVG = avg,
                          RMSE = rmse,
                          MAE = mae,
                          coeficientes = lin_reg$coefficients
)

# GUARDAR MODELO DE LIN REG
write_rds( lin_reg,'modelos/lin_reg.RDS')
# CARGAR MODELO DE LIN REG
lin_reg <- readRDS('modelos/lin_reg.RDS')

# GUARDAR DATAFRAME DE RESULTADOS LIN REG
write_rds(res_lin_reg,'data/resultados_lm.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
res_lin_reg <- readRDS('data/resultados_lm.RDS')

# GUARDAR METRICAS DE EVALUACION EN VALIDATION-SET
write_rds(metricas_lm,'data/metricas_lm.RDS')
# CARGAR DATAFRAME DE RESULTADOS LIN REG
metricas_lm <- readRDS('data/metricas_lm.RDS')



