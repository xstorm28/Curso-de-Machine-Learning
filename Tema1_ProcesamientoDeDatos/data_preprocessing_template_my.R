#Plantilla para el Pre Procesado de Datos

#Importar el Data Set
dataset = read.csv('Data.csv')

#Tratamiento de los valores NA

dataset$Age = ifelse(is.na(dataset$Age)
                     ,ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary)
                     ,ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
#Codificar las variables categoricas
#Independent
dataset$Country = factor(dataset$Country,
                         levels =c('France','Spain','Germany'),
                         labels = c(1,2,3))
#Dependent
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('Yes','No'),
                           labels = c(1,0))
# Dividir el data set en conjunto de entrenamiento y conjunto de testing 
