#height
set.seed(2510)
x <- sample(140:195, 1000, replace = T)
#weight
set.seed(99)
weight <- sample(45:100, 1000, replace = T)

lm(x ~ weight)

aa <- sample(c(1), 1000, replace = T)
a <- data.frame(aa,x)
a <- as.matrix(a)

hatb <- solve(t(a)%*%a)%*%t(a)%*%data$height
rownames(hatb) <- c("b0","b1")
colnames(hatb) <- c("X")
hatb
