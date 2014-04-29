HELPFUL LINKS:
http://stat.duke.edu/courses/Fall12/sta101.001/labs/Rcommands.pdf


#Load data from URL
load(url("http://s3.amazonaws.com/assets.datacamp.com/course/dasi/cdc.Rdata"))

#mean(), median(), and var() fns to find mean, median and variance

#Categorical data - look @ relative & absolute frequencies

#table() counts # of times (frequency) each kind of category occurs in a variable
table(cdc$genhlth)
#relative frequency = frequency / # of observations

table(cdc$genhlth) / sum(table(cdc$genhlth))

##Cool - Mosaic Plot
#See how diff. categories relate to each other. Example: How many people have smoked across each gender?
gender_smokers = table(cdc$gender, cdc$smoke100)
gender_smokers
# Plot the mosaicplot:
mosaicplot(gender_smokers)

#Lookup Index Range like rows 1-10 in column 6.
cdc[1:10, 6]
#or
cdc$weight[5]   //gets me row 5 for weight variable

***#To get dataframe that's above/lower a value or equal to a value (or just 'male' values), use subset()
new subset = subset(cdc, cdc$gender == "m")

#Sort data with many conditioning, using and '&' and or '|'. 
#Females over 30.
subset(cdc, cdc$gender == "f" & cdc$age > 30)
#Females or people over 30
subset(cdc, cdc$gender == "f" | cdc$age > 30)
#All 100+ cigarette smokers under 23
under23_and_smoke = subset(cdc, cdc$age<23 & cdc$smoke100 == 1)   //Equals is ==
  
#plot numeric data using boxplot. Give summary as well to get real info
boxplot(cdc$height)
summary(cdc$height)

#Compare two sets with '~' symbol
boxplot(cdc$weight ~ cdc$smoke100)
# Calculate the BMI:
bmi = cdc$weight / cdc$height^2 * 703
# Draw the side-by-side boxplot of bmi to general health:
boxplot(bmi ~ cdc$genhlth)

## Draw a histogram of bmi:
hist(bmi)
# And one with breaks set to 50: //breaks give me control over # of bins)
hist(bmi, breaks = 50)
# And one with breaks set to 100:
hist(bmi, breaks = 100)

##Tranform data
linear -> log
sq root
inverse


/////*****/////Lab2/////*****/////
# Load the data frame
load(url("http://s3.amazonaws.com/assets.datacamp.com/course/dasi/kobe.RData"))

#Draw barplot (make sure to put table)
barplot(table(kobe_streak))

#If I want to compare a phenomenon with a random sample, do a simulation
#Simulate flipping fair coin
outcomes = c("heads", "tails")  //Defines outcome
sample(outcomes, size=1, replace=TRUE)     //Gives me 1 outcome. Change size to give me more outcomes. Replace = TRUE , means that each new trial has the same probabilities as the 1st trial. Like no removal of sample size after each trial 

#Flip many coins and measure
#1 Run the simulation:
outcomes = c("heads", "tails")
sim_fair_coin = sample(outcomes, size=100, replace=TRUE, prob = c(0.5,0.5))   //prob is an argument that sets probability of head and tail respectively  
#2 Print the object:
sim_fair_coin
#3 Compute the counts of heads and tails:
table(sim_fair_coin)

# The 'kobe' data frame is already loaded into the workspace.  So is the
# 'sim_basket' simulation.

# Calculate streak lengths:
kobe_streak = calc_streak(kobe$basket)
sim_streak = calc_streak(sim_basket)

# Compute summaries:
summary(kobe_streak)
summary(sim_streak)

# Make bar plots:
kobe_table = table(kobe_streak)
sim_table = table(sim_streak)
barplot(kobe_table)
barplot(sim_table)



###LAB 3A###
#Load Data
load(url("http://s3.amazonaws.com/assets.datacamp.com/course/dasi/ames.RData"))
# Make some preliminary inspections:
head(ames)
tail(ames)
names(ames)

##Get a random sample from data set (ie - to calculate mean from a sample)
sample(variable, size of sample)

#For Loop
# The vector 'sample_means50' is initialized with 5,000 'NA' values (until I replace NA's with real data set)
sample_means50 = rep(NA, 5000)

# The for loop runs 5000 times, with 'i' taking values 1 up to 5000
for (i in 1:5000) {
  # Take a random sample of size 50
  samp = sample(area, 50)
  # Store the mean of the sample in the 'sample_means50' vector on the ith
  # place
  sample_means50[i] = mean(samp)
  # Print the counter 'i'
  print(i)
}

# Print the first few random medians
head(sample_means50)

###Plotting effects of diff. sample sizes
# Divide the plot in 3 rows:
par(mfrow = c(3, 1))

# Define the limits for the x-axis:
xlimits = range(sample_means10)

# Draw the histograms:
hist(sample_means10, breaks = 20, xlim = xlimits)
hist(sample_means50, breaks = 20, xlim = xlimits)
hist(sample_means100, breaks = 20, xlim = xlimits)


###LAB 3B###
#Answer the questions: "Based on this sample, what can we infer about the population?"
# Take a sample of size 60 of the population:
population = ames$Gr.Liv.Area
samp = sample(ames$Gr.Liv.Area, 60)

# Calculate the mean:
sample_mean = mean(samp)

# Draw a histogram of sample:
hist(samp)

## sample mean, usually denoted as 'x with a line over it' (here we're calling it sample_mean). sample mean is a good point estimate
## but it would be useful to show how uncertain we are of that estimate. Uncertainty measured with confidence intervals
## Here, calculate 95% confidence interval for sample mean.  
#1. Define Standard Error = (sample standard deviation) / (sq.rt sample size)
se = sd(samp)/sqrt(60)
#2. Add & Subtract 1.96 standard errors to the point estimate.
lower = sample_mean - 1.96 * se
upper = sample_mean + 1.96 * se
c(lower, upper)

#3. It is an important inference that we make with this: 
#even though we don't know what the full population looks like, 
#we're 95% confident that the true average size of houses in Ames lies between the values 'lower' and 'upper'.
# '95% confidence' means 95% of random samples of size 60 will yield conf. intervals that contain the true average area of houses in Ames, Iowa.

#Now, to show how sample means and conf. intervals vary from one sample to another, we will recreate many samples.

#1. Obtain a random sample.
#2. Calculate the sample's mean and standard deviation.
#3. Use these statistics to calculate a confidence interval.
#4. Repeat steps (1)-(3) 50 times.

# Initialize 'samp_mean', 'samp_sd' and 'n':
samp_mean = rep(NA, 50)  //Makes vector of 50 NA values.
samp_sd = rep(NA, 50)
n = 60

#Now that we've initialized our objects, we can use a for loop to calculate 50 times the mean and standard deviation of random samples.
for (i in 1:50) {
  samp = sample(population, n)       //obtain sample size n = 60 from population
  samp_mean[i] = mean(samp)         //save sample mean in ith element of samp_mean
  samp_sd[i] = sd(samp)             //save sample sd in ith element of samp_sd
}
# Calculate the interval bounds here:
lower =  samp_mean - 1.96 * (samp_sd)/(sqrt(n))
upper =  samp_mean + 1.96 * (samp_sd)/(sqrt(n))
# Plotting the confidence intervals:
pop_mean = mean(population)
plot_ci(lower, upper, pop_mean)

##CONCLUSION: We would expect 95% of the intervals to contain the true population mean.
90% = +- 1.645
95% = +- 1.96
99% = +- 2.58

###LAB 4: INFERENCE FOR NUMERICAL DATA###
#Find Summary of data. Tells me how much data is missing too!
summary(nc)
#Describe distribution of gained variable
boxplot(nc$gained)

##CLEANING UP DATA YAAAH!!!
na.omit(dataset) ##Strips dataset of NA values
#Example
# Create a clean version fo your data set:
gained_clean = na.omit(nc$gained)
# Set 'n' to the length of 'gained_clean':
n = length(gained_clean)

#Now, I can begin bootstrapping a conf. interv.
1. Take a bootstrap sample (a random sample with replacement of size equal to the original sample size) from the original sample.
2. Record the mean of this bootstrap sample.
3. Repeat steps (1) and (2) many times to build a bootstrap distribution.
4. Calculate the XX% interval using the percentile or the standard error method.
#Example
# Initialize the 'boot_means' object by getting 100 bootstrap samples and put them in boot_means:
boot_means = rep(NA, 100)
# Insert your for loop:
for (i in 1:100) {
  boot_sample = sample(gained_clean, n, replace = TRUE)
  boot_means[i] = mean(boot_sample)}
# Make a histogram of 'boot_means':
hist(boot_means)

#To bootstrap 10k times, then use inference()
# Load the 'inference' function:
load(url("http://s3.amazonaws.com/assets.datacamp.com/course/dasi/inference.Rdata"))
# Run the inference function:
#NOTE: boot_method can be "se" for standard error method. conflevel can be 0.95 for 95% conf. interval. interval can be for median, not mean.
inference(nc$gained, type = "ci", method = "simulation", conflevel = 0.9, est = "mean", 
          boot_method = "perc")

#Compare categorical and numerical data
# Draw boxplot. Put Numerical, then categorical
boxplot(nc$weight ~ nc$habit)   

#If I want to know the difference between the means of two different habits, 
#then compare them with by(). Can do this 
by(nc$weight, nc$habit, mean)

##But, to find if difference is statistically significant, then do inference().
#But, must first check for size of groups:
by(nc$weight, nc$habit, length)

##Then run inference code - explained:
1#The first argument is y, which is the response variable that we are interested in: nc$weight.
2#The second argument is the grouping variable, x, which is the explanatory variable – the grouping variable accross the levels of which we’re comparing the average value for the response variable, smokers and non-smokers: nc$habit.
3#The third argument, est, is the parameter we’re interested in: "mean" (other options are "median", or "proportion".)
4#Next we decide on the type of inference we want: a hypothesis test ("ht") or a confidence interval("ci").
5#When performing a hypothesis test, we also need to supply the null value, which in this case is 0, since the null hypothesis sets the two population means equal to each other.
6#The alternative hypothesis can be "less", "greater", or "twosided".
7#Lastly, the method of inference can be "theoretical" or "simulation" based.
inference(y = nc$weight, x = nc$habit, est = "mean", type = "ht", null = 0, 
          alternative = "twosided", method = "theoretical")

inference(y = nc$fage, x = nc$mature, est = "mean", type = "ht", null = 0, 
          alternative = "twosided", method = "theoretical")

Example:
  # Add the 'order' argument to the 'inference' function:
  inference(y = nc$weight, x = nc$habit, est = "mean", type = "ht", null = 0, 
            alternative = "twosided", method = "theoretical", order = c("smoker","nonsmoker"))

***ANOVA***
Conditions for ANOVA: Independence (w/in and b/w groups), approximate normality, equal variance
Variability Partitioning-
STEP 1: Find impact of between group vs within group Variability -
1. Sum of Squares Total (SST) - measures total variability in response variable (calculated similary to variance, except not scaled by sample size)
SST = SUM(observation - mean)^2

2. Sum of Squares Groups (SSG) - measures variability between groups - explained varaibiliy: deviation of group mean from overal mean, weighted by sample
SSG = SUM(# observations)(mean of response variable for group j - grand mean of response variable)^2
  
3. SUM of Squares Error (SSE) - measures variability within groups - unexplained variability = unexplained by group variable, due to other reasons
 SSE = SST - SSG
or
SSG + SSE = SST    //Here I can see total variance can come from unexplained variability or explained variability

STEP 2: Average variabilty from total variability
 1. Scaling by measure that incorporates sample sizes and number of groups -> degrees of freedom
  degrees of freedom associated with ANOVA:
    1. TOTAL: dfT = n - 1    //same as t-distribution
    2. GROUP: dfG = k - 1 
    3. ERROR: dfE = dfT - dfG

 2. Mean Squares: Avg. variability b/w and w/in groups, calculated by total variability (sum of squares) scaled by associated dfs
      GROUP: MSG = SSG/dfG
      ERROR: MSE = SSE/dfE

 3, F-Statistic = MSG / MSE

 4. Translate to p-value
    p-value: prob. of at least as large a ratio b/w the 'b/w' and 'w/in' group variabilities if in fact the means of all groups are equal
    Also, area under F-curve, w/ df (dfG & dfE) above observed F-stat.

Calculate ANOVA: pf(observed F-score, dfG, dfE, lower.tail = FALSE)   //always want to the upper tail

CONCLUSION: if p-value is small, reject null hypothesis: the data provides convincing evidence that at least one pair of population means are different from each other (but we can't tell which one)'
          However, if p-value is large, fail to reject null hypothesis: data doesn't provide' convincing evidence that 1 pair of population means are diff. from each other                                      ')

'R LAB on ANOVA
STEP 1:
#Since explanatory variable (class) has 4 levels, comparing (numerical response var) avg. scores across levels of class requires ANOVA
inference(y = gss$wordsum, x = gss$class, est = "mean", method = "theoretical", type = "ht", alternative = "greater")  //alternative = greater cuz F-tests are always one-sided
CONCLUSION: since p-value for F-test is low, we reject null and conclude there is evidence of at least 1 pair of means being diff.

STEP 2:

....


#######LAB 5: Inference for Categorical Data#######

#Create Subset of Data - subset()
us12 = subset(atheism, atheism$nationality == "United States" & atheism$year == "2012")
# Calculate the proportion of atheist responses:
proportion = nrow(subset(us12, response == "atheist")) / nrow(us12)
# Print the proportion:
proportion

#Calculate Inference. VERY COOL & EASY!!!
inference(us12$response, est = "proportion", type = "ci", method = "theoretical", success = "atheist")
      OR
# The subset for India for 2012:
india = subset(atheism, atheism$nationality == "India" & atheism$year == "2012")
# The analysis using the 'inference' function:
inference(india$response, est = "proportion", type = "ci", method = "theoretical", 
          success = "atheist")
#Basically inference fn above tells me how many atheists are in India vs. how many there are not atheists.

standard error: SE=√(p(1−p)/n)
margin of error for a 95% confidence interval: ME=1.96×SE =1.96 × √(p(1−p)/n). 

#Take subset of spain from atheism dataset 
#and calculate proportions of atheists w/in subset and use inference() to group by year

1# Take the 'spain' subset:
spain = subset(atheism, atheism$nationality == "Spain")
2# Calculate the proportion of atheists from the 'spain' subset:
proportion = nrow(subset(spain, response == "atheist"))/nrow(spain)
3# Use the inference function:
inference(spain$response, spain$year, est = "proportion", type = "ci", method = "theoretical", success = "atheist")





######LAB 6: Intro to Linear Regression

#Use scatterplot to show relationship b/w 2 numeric variables
#cor() tells me correlation coefficient. If cor. coeff is positive, then a positive linear correlation
correlation = cor(mlb11$runs, mlb11$at_bats)
correlation

#Manipulate linear regression line. Use scatterplot. Then draw 2 points (x1, y1) & (x2,y2). THen show value of squares
#Define 1st point
x1 = 5400
y1 = 650
#Define 2nd point
x2 = 5750
y2 = 750
#Plot linear regression
plot_ss(x = mlb11$at_bats, y = mlb11$runs, x1, y1, x2, y2, showSquares = TRUE)

#Removing points and adding leastSquares = TRUE, gets me the best fit line
plot_ss(x = mlb11$at_bats, y = mlb11$runs, showSquares = TRUE, leastSquares = TRUE)

#Also gets me best fit line (minimizes sum of squared residuals)
#model = lm(formula, data = my dataframe)
#formula takes form: y ~ x, or y is a function of x
m1 = lm(runs ~ at_bats, data = mlb11)           
m1

#Gets info for linear model
summary(m1)

RESULTS:
Call:
  #gives me fn I called
Residuals:
  #5-number summary of residuals
Coefficients:
  #1st column shows linear model's y-intercept and coefficient of x variable
  #With y-intercept and coefficient, I can write the least squares regression line for linear model:
    y = y-intercept + coefficient(homeruns)
    #Gives me conclusion: For each additional coefficient, the model predicts coefficient # more response variable (runs). 
Multiple R-squared: ***VERY COOL*** PREDICTS WHICH VARIABLE BEST EXPLAINS OUTCOME OF OTHER VARIABLE
  #shows proportion of variability in the response variable that's explained by explanatory variable.
  #For instance. if R-squared = .62, then 62% of variability in runs is explained by homeruns
  
#abline() plots line based on slope and intercept. Used to predict y at any value of x
  #Not recommended to predict x beyond range of observed data. This is called extrapolation
# Create a scatterplot:
plot(mlb11$runs ~ mlb11$at_bats)
# The linear model:
m1 = lm(runs ~ at_bats, data = mlb11)
# Plot the least squares line:
abline(m1)
  
#To assess if linear model is reliable, check:
#linearity
plot(m1$residuals ~ mlb11$at_bats)

abline(h = 0, lty = 3)
#nearly normal residuals
qqnorm(m1$residuals)
qqline(m1$residuals)

#constant variability.




######LAB 7: Multiple Linear Regression########
#Using Data Set: Data is not from experiment, but a study
load(url("http://s3.amazonaws.com/assets.datacamp.com/course/dasi/evals.RData"))

#To analyze if good looks affect teacher ratings, phrase question as:
"Is there an association between beauty and course evaluations"

#Count the number of scores below 3
sum(evals$scores < 3)

# Create a scatterplot for 'age' vs 'bty_avg':
plot(evals$bty_avg ~ evals$age)

# Create a boxplot for 'age' and 'gender':
boxplot(evals$age ~ evals$gender)

# Create a mosaic plot for 'rank' and 'gender':
mosaicplot(evals$rank ~ evals$gender)

#Use jitter() to create noise in data for each variable in diff. plots
plot(evals$score ~ jitter(evals$bty_avg))
plot(jitter(evals$score) ~ evals$bty_avg)

#see if the apparent trend in the plot is something more than natural variation
# Your initial plot:
plot(evals$score ~ jitter(evals$bty_avg))

# Construct the linear model:
m_bty = lm(evals$score ~ evals$bty_avg)

# Plot your linear model on your plot:
abline(m_bty)

#CONCLUSION: a statistically significant predictor, but may not be practically significant since the slope is very small.

#NOW, I want to know how strong the correlation b/w beauty rating of professor of lower rating profs and avg. beauty score
cor(evals$bty_avg, evals$bty_f1lower)
#Scatter Plot it too!
plot(evals$bty_avg ~ evals$bty_f1lower)

#MASTER ANALYSIS! 
#Look at the relationships between all beauty variables (columns 13 through 19)
plot(evals[, 13:19]). 
#CONCLUSION: These variables are collinear (correlated), and adding more than one of these variables
  #to the model would not add much value to the model. In this application and with these highly-correlated
  #predictors, it is reasonable to use the average beauty score as the single representative of these variables.


#NOW,add gender term to model
# Your new linear model:
m_bty_gen = lm(score ~ bty_avg + gender, data = evals)

# Study the outcome:
summary(m_bty_gen)

#What are conditions for reasonable regression?

#Tells me linear regression of 'beauty avg vs scores' for both males and females
multiLines(m_bty_gen)
#CONCLUSION: For two professors (one male and one female) who received the same beauty rating, 
    #the male professor is predicted to have the higher course evaluation score than the female.


#NOW, create new model with gender removed and replaced with rank
#Since rank has three levels (teaching, tenure track, tenured) two indicator variables are created:
    #one for tenure track and and one for tenured. Teaching is the reference level hence it doesn't show up
    #in the regression output.
# Your linear model with rank and average beauty:
m_bty_rank = lm(score ~ bty_avg + rank, data = evals)
# View the regression output:
summary(m_bty_rank)

#Analyzing coefficients
m_full = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
              cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m_full)

#CONCLUSION: coefficient associated with the ethnicity variable is 
  #0.19 points higher than minority professors, all else held constant.

#NOW, drop the variable with the highest p-value in the m_full model.
# The full model:
m_full = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m_full)

# Your new model with 1 variable dropped
m_new = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
             cls_students + cls_level + cls_credits + bty_avg, data = evals)
# View the regression output:
summary(m_new)

#NOTE: Coeff's of other variables changed cuz coeffs rely on other variables, especially during multiple regressions

#Now create a new model, where you will drop the variable that, when dropped, yields the highest improvement in the adjusted R^2.
#Once I find the variable that increases R^2 the most, then to complete the model selection,
   #we would continue removing variables one at a time until removal of another variable did not increase adjusted R-squared.
#HERE, I find that cls_profs raises R^2 the most. Then I continue w/out that variable with the next ones to find the highest R^2

# The full model:
m_full = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
              cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
#Gives me just r.squared value
summary(m_full)$adj.r.squared

# Remove rank:
m1 = lm(score ~ ethnicity + gender + language + age + cls_perc_eval + cls_students + 
          cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m1)$adj.r.squared

# Remove ethnicity:
m2 = lm(score ~ rank + gender + language + age + cls_perc_eval + 
          cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m2)$adj.r.squared

# Remove gender:
m3 = lm(score ~ rank + ethnicity + language + age + cls_perc_eval + 
          cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m3)$adj.r.squared

# Remove language
m4 = lm(score ~ rank + ethnicity + gender + age + cls_perc_eval + 
          cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m4)$adj.r.squared

# Remove age
m5 = lm(score ~ rank + ethnicity + gender + language + cls_perc_eval + 
          cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m5)$adj.r.squared


# Remove cls_perc_eval
m6 = lm(score ~ rank + ethnicity + gender + language + age + 
          cls_students + cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m6)$adj.r.squared
# Remove cls_students
m7 = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
          cls_level + cls_profs + cls_credits + bty_avg, data = evals)
summary(m7)$adj.r.squared
# Remove cls_level
m8 = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
          cls_students + cls_profs + cls_credits + bty_avg, data = evals)
summary(m8)$adj.r.squared
# Remove cls_profs
m9 = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
          cls_students + cls_level + cls_credits + bty_avg, data = evals)
summary(m9)$adj.r.squared

# Remove cls_credits
m10 = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
           cls_students + cls_level + cls_profs + bty_avg, data = evals)
summary(m10)$adj.r.squared
# Remove bty_avg
m11 = lm(score ~ rank + ethnicity + gender + language + age + cls_perc_eval + 
           cls_students + cls_level + cls_profs + cls_credits, data = evals)
summary(m11)$adj.r.squared

