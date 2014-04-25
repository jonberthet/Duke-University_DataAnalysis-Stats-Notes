//Set WD to Raw
setwd("~/Desktop/R/Scripts/Raw")

//Save excel spreadsheet as csv format

//Read Data
cbase2 <-read.csv("Investment.csv", header=TRUE)

//If data as Investments.csv (or all in 1 or 2 columns), then:
cbase2 = read.csv( "Investments.csv", sep='|')

//Get # rows
nrow(cbase2)

//give me names of variables (all header cells)
names(cbase2)names

//Find out the mode of each variable
sapply(cbase2, mode)

//Find out the class of each variable
sapply(cbase2, class)

//Convert column to different class, since I found out all data is a factor

//Convert "cbase2$raised_amount_usd" to integer. Must convert data.frame into data.matrix
cb <- data.matrix(cbase2)

//But now everything is an atomic vector. Change to data frame. //STILL CANT PLOT WELL & ANALYZE WELL
c <- as.data.frame(cb)


NEEDS WORK
***WEEK 8 DATA MUNGING*** 
1. FIND OUT HOW TO REMOVE '_' FROM VARIABLE NAMES

Making a Barplot of cbase2$raised_amount_usd
> v = cbase2$raised_amount_usd
> v1 = gsub(",", "",v)    //remove commas in #s.
> v2=subset( v1, v1 != "")
> v3=subset( v2, !is.na(v2))
> v4=as.numeric(v3)
> barplot(v4)
> v4=as.numeric(v1)
> barplot(v4)

Common Errors in Table Import
http://www.r-bloggers.com/using-r-common-errors-in-table-import/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+RBloggers+%28R+bloggers%29