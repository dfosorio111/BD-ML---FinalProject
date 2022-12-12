#Código de limpieza de datos

#Limpiar el ambiente
rm(list=ls())

setwd("~/Desktop/Big Data/Repositorios/BD-ML---FinalProject/Scripts/Samuel")

df <- read.csv("La_base_definitiva_final.csv")

require(pacman)
p_load(tidyverse, rvest, data.table, dplyr, skimr, caret, rio, 
       vtable, stargazer, ggplot2, boot, MLmetrics, lfe, 
       tidyverse, fabricatr, stargazer, Hmisc, writexl, viridis, here, GGally)

# Transformación variables

df <- df %>% mutate(log_pc=log(PERCAPITA))

df <- df %>% mutate(log_i=log(I_HOGAR))

df <- df %>% mutate(log_a=log(valor_arriendos))

df <- df %>% mutate(log_t=log(tiempo_promedio_transporte))


# Gasto e ingreso

ggplot(df, aes(x = log_i, y = log_y))+
  geom_point()+
  geom_smooth(method = "loess", level = 0.95)+
  ggtitle("Gasto en educación e ingreso del hogar")+
  xlab("Logaritmo del ingreso del hogar")+
  ylab("Logaritmo del gasto en educación (% total)")+
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5, size = 12), axis.title.x = element_text(hjust = 0.5, size = 10), axis.title.y = element_text(hjust = 0.5, size = 10), axis.text = element_text(size = 10) )

ggsave("Graficos/ingreso", dpi=300, dev='png', height=7, width=7, units="in")

# Gráfico de area barras del Max Educ

df2 <- df %>% subset(Max_educ>0)

df2$Max_educ <- factor(df2$Max_educ, order = TRUE, levels=c(1,2,3,4,5,6,7,8,9,10,11,12,13))

labels <- c("Ninguno",
             "Preescolar", 
             "Primaria", 
             "Secundaria", 
             "Media", 
             "Técnico \n incompleto", 
             "Técnico \n completo",
             "Tecnológico \n incompleto", 
             "Tecnológico \n completo",
             "Universitario \n incompleto", 
             "Universitario \n completo",
             "Postgrado \n incompleto", 
             "Postgrado \n completo")

max_educ <- df2 %>% 
  group_by(Max_educ) %>% 
  summarise(nuevo_educ = mean(Var_y,na.rm=TRUE))

ggplot(max_educ, aes(x = Max_educ, y = nuevo_educ)) + 
  geom_col(fill="lightblue")+
  ggtitle("Gasto en educación según máximo nivel educativo del hogar")+
  ylab("% del gasto en educación")+
  xlab("Máximo nivel educativo alcanzado")+
  scale_color_viridis(option = "D")+
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_x_discrete(labels=labels, guide = guide_axis(n.dodge=2)) 

ggsave("Graficos/maxeduc", dpi=300, dev='png', height=7, width=10, units="in")

ggplot(df2, aes(x = Max_educ , y = log_y)) +
  geom_boxplot(show.legend = FALSE)+
  ggtitle("Gasto en educación según máximo nivel educativo")+
  ylab("Gasto en educación (% del total)")+
  xlab("")+
  theme_classic()+
  labs(fill = "")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(text = element_text(size = 12), plot.title = element_text(hjust = 0.5, size = 14), legend.text = element_text(size = 12), legend.title = element_text(size = 12))+
  scale_x_discrete(labels=labels, guide = guide_axis(n.dodge=2)) 

# Gasto y arriendo

ggplot(df, aes(x = log_a, y = log_y))+
  geom_point()+
  geom_smooth(method = "loess", level = 0.95)+
  ggtitle("Gasto en educación y valor del arriendo")+
  xlab("Logaritmo del valor del arriendo")+
  ylab("Logaritmo del gasto en educación (% total)")+
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5, size = 12), axis.title.x = element_text(hjust = 0.5, size = 10), axis.title.y = element_text(hjust = 0.5, size = 10), axis.text = element_text(size = 10) )

ggsave("Graficos/arriendo", dpi=300, dev='png', height=7, width=7, units="in")

# Gasto y tiempo

df$transporte <- factor(df$transporte, order = FALSE, levels=c(1,2,3,4,5,6,7,8))

labels3 <- c("Vehículo particular",
            "Transporte escolar", 
            "Transporte público", 
            "A pie", 
            "Bicicleta", 
            "Caballo o mula", 
            "Lancha, planchón \n o canoa",
            "Otro")

ggplot(df, aes(x = log_t, y = log_y, colour= transporte))+
  geom_point(alpha = 0.1)+
  geom_smooth(method="lm",se = FALSE)+
  ggtitle("Gasto en educación y tiempo de transporte")+
  xlab("Logaritmo del tiempo de transporte")+
  ylab("Logaritmo del gasto en educación (% total)")+
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5, size = 12), axis.title.x = element_text(hjust = 0.5, size = 10), axis.title.y = element_text(hjust = 0.5, size = 10), axis.text = element_text(size = 10) )+
  scale_colour_discrete(name = "Medio de transporte", label=labels3)

ggsave("Graficos/tiempo", dpi=300, dev='png', height=7, width=7, units="in")

# Gasto y estudiantes

ggplot(df, aes(x = Estudiante._10_años, y = log_y))+
  geom_point()+
  geom_smooth(method = "loess", level = 0.95)+
  ggtitle("Gasto en educación y proporción de estudiantes")+
  xlab("Proporción de estudiantes")+
  ylab("Logaritmo del gasto en educación (% total)")+
  theme_classic()+
  theme(plot.title = element_text(hjust = 0.5, size = 12), axis.title.x = element_text(hjust = 0.5, size = 10), axis.title.y = element_text(hjust = 0.5, size = 10), axis.text = element_text(size = 10) )

ggsave("Graficos/estudiantes", dpi=300, dev='png', height=7, width=7, units="in")


# Gasto y pobreza

df$P5230 <- factor(df$P5230, order = FALSE, levels=c(1,2))

labels2 <- c("Sí",
            "No")

ggplot(df, aes(x = P5230 , y = log_y)) +
  geom_boxplot(show.legend = FALSE)+
  ggtitle("Gasto en educación según pobreza")+
  ylab("Gasto en educación (% total)")+
  xlab("¿Se considera pobre?")+
  theme_classic()+
  labs(fill = "")+
  theme(plot.title = element_text(hjust = 0.5, size = 12), axis.title.x = element_text(hjust = 0.5, size = 10), axis.title.y = element_text(hjust = 0.5, size = 10), axis.text = element_text(size = 10) )+
  scale_x_discrete(labels=labels2, guide = guide_axis(n.dodge=1)) 

ggsave("Graficos/pobreza", dpi=300, dev='png', height=7, width=7, units="in")

# Gasto y computadore

df$P1077S22 <- factor(df$P1077S22, order = FALSE, levels=c(1,2))

ggplot(df, aes(x = P1077S22 , y = log_y)) +
  geom_boxplot(show.legend = FALSE)+
  ggtitle("Gasto en educación según computador")+
  ylab("Gasto en educación (% total)")+
  xlab("¿Tiene computador?")+
  theme_classic()+
  labs(fill = "")+
  theme(plot.title = element_text(hjust = 0.5, size = 12), axis.title.x = element_text(hjust = 0.5, size = 10), axis.title.y = element_text(hjust = 0.5, size = 10), axis.text = element_text(size = 10) )+
  scale_x_discrete(labels=labels2, guide = guide_axis(n.dodge=1)) 

ggsave("Graficos/computador", dpi=300, dev='png', height=7, width=7, units="in")
