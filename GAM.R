library(xlsx)
library(mgcv)

df = read.xlsx2('C:\\Users\\bickels\\Documents\\GitHub\\geoworm\\data\\occ.xlsx',1,colClasses='numeric')
#df$occ = as.numeric(df$occ)

gam(occ~s(PL),data=df)
