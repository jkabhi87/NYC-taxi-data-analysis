# NYC-taxi-data-analysis
Predicting trip durations for NYC taxi dataset

I first started with the task of determining the features that will have larger impact on the trip duration. The obvious ones being pickup hour and distance. I dropped a lot of features that were either not useful or were redundant. For example, transaction id has no impact on the trip duration.
Similarly, I removed drop off date time as i can calculate this value using the pickup time and trip duration. I also mapped the trip duration against the passenger count and it was pretty much even for all the passenger count.
Since pickup hour and distance were not readily available in the data that was provided, I modified the csv files to separate the hour from the pickup datetime.
I also have a calculation in the excel that considers the minute of the pick in date time and calculates the hours past midnight which takes into account the minutes along with the hour of the pickup. This is a floating point value.
I also calculate the square distance (euclidean distance in my python program based on the pick up and drop off locations).
I have written another python program that groups the data points based on various features against the average trip duration.
Here, i found that Storeandforward had some impact on the trip duration as they trips with this flag set to N had longer trip duration.
So my final data points are defined by the following features: pickup hour, vendor id, store and forward flag, square distance and manhattan distance.
The reason I thought of using both distance metrics is because, distance is the most important feature in the model that determines the trip duration.

Observations:
I started out with the simple model that used most of the original features and the RMSE calculated was enormous. So i decided to do some feature engineering to improve the model. 
But looks like even with the feature engineering, there is only so much we can improve. I even looked at the discussion thread in kaggle and people have gotten better results only after integrating the weather and traffic data along with the provided dataset.
The best performance for my model was when i could see the RMSE in the range of around 3800-4100 and MAE in the range of 350-430.

Cross validation:
I struggled to work with the standard cross validation library in spark. I particularly found the gird concept to be a little confusing.
So, i wrote my own cross validation program that splits the training set into 10 parts. Then i trained the model 10 times keeping a different part of the training set for validation and using the rest for training.
Then i calculated the average of RMSE and MAE values for all 10 runs.
The only drawback with writing my own CV program was that the execution took a long time. close to an hour in most cases.
I ran cross validation for all the models. I found that SGD worked best when i used the step size to use exactly one record. 
I also tried running CV with multiple combinations of iterations and step. I tried the values of 10,100,200,1000, 2000, 5000,10000 and 120000 for iterations.
While i used 0.01, 0.0001, 0.00000085938265 for step. I found the least error values for the combination of 10000 for iterations and 0.00000085938265 for step size for SGD (which corresponds to 1 data point in training data) and iterations = 100 for GD. I used these values for my submission.

Regularization:
I have used LassoWithSGD for L1 regularization while RidgeRegressionWithSGD for L2 regularization. The model even in its best performance, underfits the data. So, regularization didn't really help me improve the performance.
As the results show, it performs almost on par with the regular models.

Performed the following tasks:
1. wrote a program to do the CV on the training data set.
2. wrote programs to calculate RMSE and MAE values for  gradient descent models using LinearRegressionWithSGD, RidgeRegressionWithSGD, LassoWithSGD with minibatchsize set to 1.0.
3. wrote programs to calculate RMSE and MAE values for stochastic gradient models using LinearRegressionWithSGD, RidgeRegressionWithSGD, LassoWithSGD.

All tasks have been implemented in python.
1. The zipped file contains one folder which contains 3 folders : Code, Pseuodocode  and Output
2. The output folder contains the output files and the graphs
3. The folder named "Code" has 4 subfolders
    a. The folder CrossValidation contains python script for cross validation.
    b. The folder GD_Regression contains python scripts for regular GD, GD with L1 and GD with L2.
    c. The folder SGD_Regression contains the python script for regular SGD, SGD with L1 and SGD with L2.
    d. The folder Extra contains the python script that was used to plot features against average trip duration to determine feature importance.
3. The folder named "Pseudocode" has 3 subfolders
    a. The folder CrossValidation contains pseudocode for the python script for cross validation.
    b. The folder GD_Regression contains pseudocode for the python scripts for regular GD, GD with L1 and GD with L2.
    c. The folder SGD_Regression contains the pseudocode for the python script for regular SGD, SGD with L1 and SGD with L2.

Conclusion: There is a limit as to how far we can reduce the RMSE and MAE for the given data set without integrating the traffic, weather and other data sets.
The training times for GD and SGD were comparable. In most cases, SGD completed the training faster than the GD. This is expected because, GD uses all the data points for each iteration in the training while SGD uses only the number of data points specified in the step size. Since this is small, SGD trains faster than GD.
