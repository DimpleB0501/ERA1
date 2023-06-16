# Question
PART 1[250]: Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github. <br/>
Use exactly the same values for all variables as used in the class <br/>
Take a screenshot, and show that screenshot in the readme file <br/>
The Excel file must be there for us to cross-check the image shown on readme (no image = no score) <br/>
Explain each major step <br/>
Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] <br/>
Upload all this to GitHub and then write all the above as part 1 of your README.md file. <br/>
Submit details to S4 - Assignment QnA. <br/>

# Solution
In session 3 we worked on writing a simple neural network in excel sheet. <br/>
We derived the formulas for forward and backward propogation using partial derviates. Following are the derived formulas <br/>
![formulas](images/formula.PNG)
<br/>
We started with the weights displayed in the figure. <br/>
At each step we updated the weights using the learning rate. The formula for weight update is <br/>`weight_new = weight_old - (learning_rate x ꝺE/ꝺweight_old)` <br/>
Following [excel sheet](https://github.com/DimpleB0501/eva8/blob/main/Session_3/nn_excel.ods) contains my work on building a simple neural network for the Assignment 3 part 1. Following figure displays the screenshot of the excel sheet <br/> 
![sc](images/part1.PNG)

2. Graphs of the learning rate <br/>
- Learning rate= 0.1 <br/>
![lr1](images/lr_0.1.PNG)

- Learning rate= 0.2 <br/>
![lr1](images/lr_0.2.PNG)

- Learning rate= 0.5 <br/>
![lr1](images/lr_0.5.PNG)

- Learning rate= 0.8 <br/>
![lr1](images/lr_0.8.PNG)

- Learning rate= 1 <br/>
![lr1](images/lr_1.PNG)

- Learning rate= 2 <br/>
![lr1](images/lr_2.PNG)
