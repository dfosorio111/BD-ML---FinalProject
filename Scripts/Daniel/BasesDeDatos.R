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


#Base educación
#Cargar base de educación
educ <- read.csv("Educación.csv")
#103 variables y 238.888 observaciones
#Lo anterior significa que las variables NO están a nivel hogar:

#Máximo nivel educativo en el hogar.P8587
#Ver los niveles educativos
educ%>%count(P8587)
#cambiar los NA por 0
educ$P8587[which(is.na(educ$P8587))] <- 0
#Ver de nuevo los niveles educativos
educ%>%count(P8587)

#Esto prácticamente está ordenado, se trata de escoger el máximo
max_educ_level <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(Max_educ = max(P8587))

#Alguien estudia en el hogar?
educ%>%count(P8586) #Bien porque no hay NA

#Como 1 es que sí estudia y 2 que no, queremos el mínimo de cada hogar
alguien_estudia <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(alguien_estudia = min(P8586))

#Tiempo promedio de desplazamiento hacia donde estudia
tiempo_promedio_transporte <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(tiempo_promedio_transporte = mean(P6167, na.rm = TRUE))

#Se crea la función de la moda
moda <- function(codes){
  which.max(tabulate(codes))
}

moda(educ$P4693)

#Principal medio de transporte hacia la institución educativa
medio_transporte <-  educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(transporte = moda(P4693))


#Tiempo dedicado al estudio (moda, máximo y mínimo)
tiempo_estudio <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(tiempo_moda_estudio = moda(P3340),
                                                                       tiempo_max_estudio = max(P3340, na.rm = TRUE),
                                                                       tiempo_min_estudio = min(P3340, na.rm = TRUE))



#Se obtiene el gasto total en educación
#Se agrupa el total gastado a nivel hogar por división (para la de educación)
Valores_nivel_hogar <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%
  summarise(valor_matricula = sum(P3341S1, na.rm = TRUE),
            valor_uniformes = sum(P3342S1, na.rm = TRUE),
            valor_utiles    = sum(P3343S1, na.rm = TRUE),
            valor_pension   = sum(P3344S1, na.rm = TRUE),
            valor_transporte= sum(P3345S1, na.rm = TRUE),
            valor_alimento  = sum(P3346S1, na.rm = TRUE))%>%
  ungroup()


##########Tener en cuenta que la encuesta indica que es durante el AÑO ESCOLAR

#Se suma el total gastado por hogar en educación
Valores_nivel_hogar$valor_total <- apply(Valores_nivel_hogar%>%select(-DIRECTORIO,-SECUENCIA_P), 1, sum)


#Ver si los hogares recibieron becas. Como 1 es sí y 2 es no, se busca el mínimo. (Al menos alguien que estudia recibió una beca)
becas <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(beca = min(P8610, na.rm = TRUE), monto_beca = sum(P8610S1, na.rm = TRUE))
becas$beca[which(becas$beca > 2)] <- 0

#Subsidios
subsidio <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(subsidio = min(P8612, na.rm = TRUE), monto_subsidio = sum(P8612S1, na.rm = TRUE))
subsidio$subsidio[which(subsidio$subsidio > 2)] <- 0

#Crédito educativo
credito <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(credito = min(P8614, na.rm = TRUE), monto_credito = sum(P8614S1, na.rm = TRUE))
credito$credito[which(credito$credito > 2)] <- 0


################Bases de actividades extracurriculares
extracurriculares <- educ%>%group_by(DIRECTORIO, SECUENCIA_P)%>%summarise(arte = max(P3004S1, na.rm = TRUE),
                                                                          ciencia = max(P3004S2, na.rm = TRUE),
                                                                          deportes = max(P3004S3, na.rm = TRUE),
                                                                          grupos_estudio = max(P3004S4, na.rm = TRUE),
                                                                          parque = max(P3004S5, na.rm = TRUE),
                                                                          lectura = max(P3004S6, na.rm = TRUE),
                                                                          juegos = max(P3004S7, na.rm = TRUE))
#Se llevan a 0 los -infinitos
extracurriculares$arte[which(extracurriculares$arte < 0)] <- 0
extracurriculares$ciencia[which(extracurriculares$ciencia < 0)] <- 0
extracurriculares$deportes[which(extracurriculares$deportes < 0)] <- 0
extracurriculares$grupos_estudio[which(extracurriculares$grupos_estudio < 0)] <- 0
extracurriculares$parque[which(extracurriculares$parque < 0)] <- 0
extracurriculares$lectura[which(extracurriculares$lectura < 0)] <- 0
extracurriculares$juegos[which(extracurriculares$juegos < 0)] <- 0

extracurriculares$alguna_extra <- apply(extracurriculares%>%ungroup()%>%select(-DIRECTORIO,-SECUENCIA_P), 1, max)
#Ojo porque si se mandan por separado y se manda esta última se crea multicolinealidad en una regresión


#Unión de las bases creadas hasta el momento.
base_educ <- full_join(max_educ_level, alguien_estudia)
base_educ <- full_join(base_educ, medio_transporte)
base_educ <- full_join(base_educ, tiempo_promedio_transporte)
base_educ <- full_join(base_educ, tiempo_estudio)
base_educ <- full_join(base_educ, Valores_nivel_hogar)
base_educ <- full_join(base_educ, becas)
base_educ <- full_join(base_educ, subsidio)
base_educ <- full_join(base_educ, credito)
base_educ <- full_join(base_educ, extracurriculares)

#Si nadie estudia, las siguientes variables se llevan a 0, 
#creería que estas observaciones igual deben eliminarse
base_educ$transporte[which(base_educ$alguien_estudia == 2)] <- 0
base_educ$tiempo_moda_estudio[which(base_educ$alguien_estudia == 2)] <- 0
base_educ$tiempo_max_estudio[which(base_educ$alguien_estudia == 2)] <- 0
base_educ$tiempo_min_estudio[which(base_educ$alguien_estudia == 2)] <- 0

#Los inf y los -inf se llevan a NaN
base_educ$tiempo_max_estudio[which(base_educ$tiempo_max_estudio < 0)] <- NaN
base_educ$tiempo_min_estudio[which(base_educ$tiempo_min_estudio > 10000)] <- NaN

#Suma de todas las ayudas existentes


base_educ$ayudas_total <- apply(base_educ%>%ungroup()%>%select(monto_beca, monto_subsidio, monto_credito), 1, sum)



#Exportar la base de datos.
#directorio Daniel  
setwd("C:/Users/danie/OneDrive/Escritorio/Uniandes/PEG/Big Data and Machine Learning/BD-ML---FinalProject/Data_definitiva")
write.csv(base_educ, "Base_educacion.csv")
