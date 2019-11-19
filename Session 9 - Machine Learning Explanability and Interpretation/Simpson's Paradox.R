# Load the tidyverse
library(tidyverse)

# Generating correlated data with mvrnorm() from the MASS library
library(MASS)

# Sample Means
mu <- c(20,4)

# Define our covariance matrix, and specify the covariance relationship (i.e. 0.7 in this case)
Sigma <- matrix(.7, nrow=2, ncol=2) + diag(2)*.3

# create both variables with 100 samples
vars <- mvrnorm(n=100, mu=mu, Sigma=Sigma)

# Examine the data and the correlation
head(vars)
cor(vars)

# Plot the variables
plot(vars[,1],vars[,2])

# Create a function for generating 2 correlated variables given variable means
corVars<-function(m1,m2,confVar){
  mu <- c(m1,m2)
  Sigma <- matrix(.7, nrow=2, ncol=2) + diag(2)*.5
  vars <- mvrnorm(n=100, mu=mu, Sigma=Sigma)
  Var1<-vars[,1]
  Var2<-vars[,2]
  df<-as.data.frame(cbind(Var1 = Var1,Var2 = Var2,Var3 = confVar))
  df$Var1<-as.numeric(as.character(df$Var1))
  df$Var2<-as.numeric(as.character(df$Var2))
  df
}

# Re-running for multiple sets and combining into a single dataframe df
d1 <- corVars(m1 = 20, m2 = 82, confVar = "Algebra")
d2 <- corVars(m1 = 18, m2 = 84, confVar = "English")
d3 <- corVars(m1 = 16, m2 = 86, confVar = "Social Studies")
d4 <- corVars(m1 = 14, m2 = 88, confVar = "Art")
d5 <- corVars(m1 = 12, m2 = 90, confVar = "Physical Education")

# Create the aggregate data
df<-rbind(d1,d2,d3,d4,d5)

# Grade & Study Time Plot
df %>%
  ggplot(aes(x = Var1, y = Var2/100)) +
  geom_jitter(aes(size = 13), alpha = 0.55, shape = 21, fill = "darkgray", color = "black") +
  scale_y_continuous(name = "Final Percentage")+
  scale_x_continuous(name = "Approximate Hours for Preparation")+
  guides(size = FALSE) +
  ggtitle("Impact of Studying on Final Grades")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme_bw()

# Grade & Study Time Correlation
cor(df$Var1, df$Var2)

# PhysEd Plot
df %>% 
  filter(Var3 == 'Physical Education') %>%
  ggplot(aes(x = Var1, y = Var2/100)) +
  geom_jitter(aes(size = 13), alpha = 0.55, shape = 21, fill = "darkgray", color = "black") +
  scale_y_continuous(name = "Final Percentage")+
  scale_x_continuous(name = "Approximate Hours for Preparation")+
  guides(size = FALSE) +
  ggtitle("Impact of Studying on Final Grades (Physical Education Only)")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme_bw()

# PhysEd Correlation
cor(df$Var1[df$Var3 == 'Physical Education'], df$Var2[df$Var3 == 'Physical Education'])

# Confounding plot
df %>%
  ggplot(aes(x = Var1, y = Var2/100)) +
  geom_jitter(aes(size = 1, fill = Var3), alpha = 0.25, shape = 21) +
  guides(fill = guide_legend(title = "Course Class", override.aes = list(size = 5)),
         size = FALSE) +
  scale_y_continuous(name = "Testing Results")+
  scale_x_continuous(name = "Approximate Hours for Preparation")+
  ggtitle("Impact of Studying on Final Grades")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme_bw()

