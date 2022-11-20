
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
       glmnet, janitor, doParallel, rattle, fastDummies, tidymodels, themis, AER, randomForest,xgboost, ranger)

#directorio Daniel  
setwd("C:/Users/danie/OneDrive/Escritorio/Uniandes/PEG/Big Data and Machine Learning/BD-ML---FinalProject/Data")

#URL del diccionario de datos
browseURL("https://microdatos.dane.gov.co//catalog/734/get_microdata")

#Diccionario de Ignacio
browseURL("https://ignaciomsarmiento.github.io/GEIH2018_sample/dictionary.html")

#Cargar base de características y composición del hogar
caract_hogar <- read.csv("Caracteristicas y composicion del hogar.csv", sep = ";") #Parece estar a nivel individuo, pero son características de hogar
#52 variables y 257.589 observaciones

#Cargar base de educación
educ <- read.csv("Educación.csv")
#103 variables y 238.888 observaciones

#Condiciones de vida del hogar y tenencia de bienes
cond_hogar <- read.csv("Condiciones de vida del hogar y tenencia de bienes.csv")
#118 variables con 89.203 observaciones

#Se carga la base "servicios del hogar". En esta base se encuentran los ingresos
servicios_hogar <- read.csv("Servicios del hogar.csv")

#Atención integral niños y niñas menores de 5 años
de_cero_a_siempre <- read.csv("Cero_a_cinco.csv", sep = ";")


#Se agrupa el total gastado a nivel hogar por división (para la de educación)
Valores_nivel_hogar <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%
  summarise(valor_matricula = sum(P3341S1, na.rm = TRUE),
            valor_uniformes = sum(P3342S1, na.rm = TRUE),
            valor_utiles    = sum(P3343S1, na.rm = TRUE),
            valor_pension   = sum(P3344S1, na.rm = TRUE),
            valor_transporte= sum(P3345S1, na.rm = TRUE),
            valor_alimento  = sum(P3346S1, na.rm = TRUE))%>%
  ungroup()

#Se suma el total gastado por hogar en educación
Valores_nivel_hogar$valor_total <- apply(Valores_nivel_hogar%>%select(-DIRECTORIO,-SECUENCIA_P), 1, sum)



#Se agrupa el total gastado a nivel hogar por división (para la de cero a siempre)
Valores_nivel_hogar_5 <- de_cero_a_siempre%>%group_by(DIRECTORIO, SECUENCIA_P)%>%
  summarise(valor_matricula = 12*sum(P6169S1, na.rm = TRUE),
            valor_uniformes = sum(P8564S1, na.rm = TRUE),
            valor_utiles    = sum(P8566S1, na.rm = TRUE),
            valor_utiles2   = sum(P8568S1, na.rm = TRUE),
            valor_pension   = 12*sum(P6191S1, na.rm = TRUE),
            valor_transporte= 12*sum(P8572S1, na.rm = TRUE),
            valor_alimento  = 12*sum(P8574S1, na.rm = TRUE))%>%
  ungroup()%>%
  mutate(valor_matricula = ifelse(valor_matricula == 99, 0, valor_matricula),
            valor_uniformes = ifelse(valor_uniformes == 99, 0, valor_uniformes),
            valor_utiles    = ifelse(valor_utiles == 99, 0, valor_utiles),
            valor_utiles2   = ifelse(valor_utiles2 == 99, 0, valor_utiles2),
            valor_pension   = ifelse(valor_pension == 99, 0, valor_pension),
            valor_transporte= ifelse(valor_transporte == 99, 0, valor_transporte),
            valor_alimento  = ifelse(valor_alimento == 99, 0, valor_alimento))

#Se suma el total gastado por hogar en educación
Valores_nivel_hogar_5$valor_total <- apply(Valores_nivel_hogar_5%>%select(-DIRECTORIO,-SECUENCIA_P), 1, sum)


#Unir servicios del hogar con valores_nivel_hogar (Se unen por DIRECTORIO Y SECUENCIA)
Union_servicios_valores <- full_join(servicios_hogar, Valores_nivel_hogar)


#Unir servicios del hogar con valores_nivel_hogar_5 (Se unen por DIRECTORIO Y SECUENCIA)
Union_servicios_valores_5 <- left_join(Valores_nivel_hogar_5, servicios_hogar)
Union_servicios_valores_5$paga_algo <- ifelse(Union_servicios_valores_5$valor_total>0,1,0)
Union_servicios_valores_5%>%count(paga_algo)

Union_servicios_valores_5 <- Union_servicios_valores_5%>%mutate(Variable_y = 100*valor_total/(12*I_HOGAR))

quantile(PRUEBA$Variable_y, probs = seq(0,1,0.005))

PRUEBA <- Union_servicios_valores_5%>%subset(valor_total>0)%>%subset(!is.na(I_HOGAR)&I_HOGAR>0)

PRUEBA2 <- PRUEBA%>%subset(Variable_y > 0.002 & Variable_y < 90)


hist(PRUEBA2$Variable_y)

summary(PRUEBA$Variable_y)

View(PRUEBA2%>%select(I_HOGAR, valor_matricula, valor_total, Variable_y))

