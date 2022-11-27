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
setwd("C:/Users/danie/OneDrive/Escritorio/Uniandes/PEG/Big Data and Machine Learning/BD-ML---FinalProject/Data")

#URL del diccionario de datos
browseURL("https://microdatos.dane.gov.co//catalog/734/get_microdata")


#Descargar bases Daniel
#1. Atención integral niños y niñas menores de 5 años
de_cero_a_siempre <- read.csv("Cero_a_cinco.csv", sep = ";")

#Esta base tiene 63 variables y 18701 observaciones
#En esta base no hay ningún jefe de hogar
#Vamos a dejar por ahora esta base de lado:

#2. tenencia y financiación de vivienda
Tenencia_vivienda <- read.csv("tenencia_financiacion.csv", sep = ";")
#39 variables y 89203 observaciones

#Variables que se quedan
#P5095 Vivienda propia, en arriendo, etc.
#P5130 si tuviera que pagar arriendo por esta vivienda, cuánto estima que pagaría?
#p5140 Llena los NA de la 5130 teniendo esto en mente, se forma la siguiente variable:
Tenencia_vivienda <- Tenencia_vivienda%>%mutate(valor_arriendos = ifelse(is.na(P5130), P5140, P5130))
#Acá toca tener ojo con una cosa. Hay muchos 99, que aunque no los cuentan como NA, practicamente lo son
sum(is.na(Tenencia_vivienda$valor_arriendos))
sum(Tenencia_vivienda$valor_arriendos == 99) #Toca pensar qué hacer con esos, porque alteran todo
#Puede que al final esta variable no importe mucho, si se tiene el ingreso del hogar.

Tenencia_vivienda <- Tenencia_vivienda%>%select(DIRECTORIO, SECUENCIA_ENCUESTA, SECUENCIA_P, ORDEN,
                                                FEX_C, P5095, valor_arriendos)
#Exportar la base de datos.
#directorio Daniel  
setwd("C:/Users/danie/OneDrive/Escritorio/Uniandes/PEG/Big Data and Machine Learning/BD-ML---FinalProject/Data_definitiva")
write.csv(Tenencia_vivienda, "tenencia_financiacion.csv")

Tenencia_vivienda2 <- read.csv("tenencia_financiacion_escritura.csv", sep = ";")
#Esta solo tienen una variable que no veo que sirva para mucho.

#Gastos de los hogares, lugares de compra
gastos_lugares <- read.csv("Gastos de los hogares (lugares de compra).csv", sep = ";")
#Sinceramente  no veo que las variables de aquí agregen mucho.

cond_vida <- read.csv("Condiciones de vida del hogar y tenencia de bienes.csv")
#Tiene 118 variables
#Variables que me parecen relevantes
#P5230 usted se considera pobre?
#p9090 (los ingresos del hogar alcanzan?)
#p784 algun miembro del hogar recibió subsidios?
#p784s1 (familias en acción)
#p784s1a1 (cuántos miembros recibieron familias en acción)
#p784s1a2 (cuánto recibieron de familias en acción)
#p1077s1-2-3-4-5-6-7-8-9-10-14-15-16-17-19-21-22-23 electrodomésticos y similares
#Conexión a internet: p1075 p1075s1 p1075s2
#Tiempo caminando hasta establecimiento educativo: P1913S2
#P3202S6 P3202S7 por el covid 19 atrasos en pagos educativos
#P3203S8 P3203S9 RETIROS DE EDUCACIÓN POR COVID
#P3353 (SITUACIÓN COMPARADA CON 12 MESES ATRÁS)
#p3354 (cómo ve la situación futura)

cond_vida <- cond_vida%>%select(DIRECTORIO, SECUENCIA_ENCUESTA, SECUENCIA_P, ORDEN,
                                P5230, P9090, P784, P784S1, P784S1A1, P784S1A2,
                                P1077S1, P1077S2, P1077S3, P1077S4, P1077S5, P1077S6, P1077S7,
                                P1077S8, P1077S9, P1077S10, P1077S14, P1077S15, P1077S16, P1077S17,
                                P1077S19, P1077S21, P1077S22, P1077S23, P1075,
                                P1075S1, P1075S2, P1913S2, P3202S6, P3202S7,
                                P3203S8, P3203S9, P3353, P3354)
#Exportar
write.csv(cond_vida, "condiciones_de_vida.csv")
