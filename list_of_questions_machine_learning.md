# Machine Learning - List of questions

## Learning Theory

1. Describe bias and variance with examples. 
    - ### Bias
        - Bias is the error due to overly simplistic assumptions in the learning algorithm. High bias can cause the model to miss the relevant relations between features and target outputs (underfitting), meaning the model is not complex enough to capture the underlying patterns of the data.
    - #### High Bias Example in Machine Learning
        - Imagine you are using a linear regression model to predict housing prices based on features such as square footage and the number of bedrooms. However, the relationship between these features and the price is actually non-linear. By assuming a linear relationship, your model is too simple to capture the complexities of the real-world function that determines housing prices, leading to systematic errors in predictions.
    - ### Variance
        - Variance refers to the error due to too much complexity in the learning algorithm. High variance can cause the model to model the random noise in the training data (overfitting), meaning the model learns patterns from the training data that don't generalize to unseen data.
    - #### High Variance Example in Machine Learning
        - Consider a decision tree model that you've allowed to grow without constraints until each leaf node represents only one training example. Such a model might perform perfectly on the training set, capturing every detail (including noise) in the dataset. However, this complexity makes the model highly sensitive to the specifics of the training data, and it is likely to perform poorly on new, unseen data because it has learned the noise as if it were a real pattern.



1. What is Empirical Risk Minimization?
    ### Empirical Risk Minimization (ERM)

    Empirical Risk Minimization (ERM) is a fundamental principle in statistical learning and machine learning that aims at minimizing the loss (or error) on the training dataset to identify the best model. The essence of ERM is to select the model or model parameters that reduce the empirical risk, essentially the average loss over the training samples.

    ### How Does ERM Work?

    The empirical risk is calculated by applying a loss function to the model's predictions on the training data and then averaging these losses. The loss function measures the discrepancy between the predicted and actual values. For regression tasks, mean squared error is a common loss function, and for classification tasks, cross-entropy loss is often used.

    ERM's goal is to find the model parameters that minimize this empirical risk. However, minimizing empirical risk directly can lead to overfitting, especially if the model is too complex, perfectly fitting the training data, including its noise. This happens because ERM focuses solely on the training data, neglecting how the model performs on unseen data.

    ### Mathematical Formulation

    Given a dataset \(\{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}\) where \(x_i\) represents the input features and \(y_i\) are the labels, and a model \(f\) parameterized by \(\theta\), the empirical risk \(R_{emp}(\theta)\) is defined as:

    \[R_{emp}(\theta) = \frac{1}{n} \sum_{i=1}^{n} L(f(x_i; \theta), y_i)\]

    Here, \(L\) is the loss function, \(n\) is the number of samples in the training dataset, \(f(x_i; \theta)\) denotes the model's prediction for input \(x_i\), and \(y_i\) are the true labels.

    ### Example

    Consider a linear regression problem where the goal is to predict a person's weight based on their height. The model is a linear function \(f(x) = mx + b\), with \(x\) representing height, \(m\) and \(b\) as parameters to be learned, and \(f(x)\) as the predicted weight. Using mean squared error for the loss function, ERM will determine the values of \(m\) and \(b\) that minimize the average squared difference between the predicted and actual weights in the training dataset.

    ### Challenges and Solutions

    While ERM is a potent principle, direct application can result in overfitting, particularly with complex models and limited data. To mitigate this, several approaches are employed, such as regularization (adding a penalty to the loss to discourage complexity) and validation methods (dividing the data into training and validation sets to gauge model performance on unseen data).

    ERM underpins many machine learning algorithms, guiding the optimization process towards models that exhibit good performance on the provided data.

1. What is Union bound and Hoeffding's inequality? 
1. Write the formulae for training error and generalization error. Point out the differences.
1. State the uniform convergence theorem and derive it. 
1. What is sample complexity bound of uniform convergence theorem? 
1. What is error bound of uniform convergence theorem? 
1. What is the bias-variance trade-off theorem? 
1. From the bias-variance trade-off, can you derive the bound on training set size?
1. What is the VC dimension? 
1. What does the training set size depend on for a finite and infinite hypothesis set? Compare and contrast. 
1. What is the VC dimension for an n-dimensional linear classifier? 
1. How is the VC dimension of a SVM bounded although it is projected to an infinite dimension? 
1. Considering that Empirical Risk Minimization is a NP-hard problem, how does logistic regression and SVM loss work? 

## Model and feature selection
1. Why are model selection methods needed?
1. How do you do a trade-off between bias and variance?
1. What are the different attributes that can be selected by model selection methods?
1. Why is cross-validation required?
1. Describe different cross-validation techniques.
1. What is hold-out cross validation? What are its advantages and disadvantages?
1. What is k-fold cross validation? What are its advantages and disadvantages?
1. What is leave-one-out cross validation? What are its advantages and disadvantages?
1. Why is feature selection required?
1. Describe some feature selection methods.
1. What is forward feature selection method? What are its advantages and disadvantages?
1. What is backward feature selection method? What are its advantages and disadvantages?
1. What is filter feature selection method and describe two of them?
1. What is mutual information and KL divergence?
1. Describe KL divergence intuitively.

## Curse of dimensionality 
1. Describe the curse of dimensionality with examples.
1. What is local constancy or smoothness prior or regularization?

## Universal approximation of neural networks
1. State the universal approximation theorem? What is the technique used to prove that?
1. What is a Borel measurable function?
1. Given the universal approximation theorem, why can't a MLP still reach a arbitrarily small positive error?

## Deep Learning motivation
1. What is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?
1. In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be recognized in the function space?
1. What are the reasons for choosing a deep model as opposed to shallow model? (1. Number of regions O(2^k) vs O(k) where k is the number of training examples 2. # linear regions carved out in the function space depends exponentially on the depth. )
1. How Deep Learning tackles the curse of dimensionality? 

## Support Vector Machine
1. How can the SVM optimization function be derived from the logistic regression optimization function?
1. What is a large margin classifier?
1. Why SVM is an example of a large margin classifier?
1. SVM being a large margin classifier, is it influenced by outliers? (Yes, if C is large, otherwise not)
1. What is the role of C in SVM?
1. In SVM, what is the angle between the decision boundary and theta?
1. What is the mathematical intuition of a large margin classifier?
1. What is a kernel in SVM? Why do we use kernels in SVM?
1. What is a similarity function in SVM? Why it is named so?
1. How are the landmarks initially chosen in an SVM? How many and where?
1. Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
1. What is the difference between logistic regression and SVM without a kernel? (Only in implementation – one is much more efficient and has good optimization packages)
1. How does the SVM parameter C affect the bias/variance trade off? (Remember C = 1/lambda; lambda increases means variance decreases)
1. How does the SVM kernel parameter sigma^2 affect the bias/variance trade off?
1. Can any similarity function be used for SVM? (No, have to satisfy Mercer’s theorem)
1. Logistic regression vs. SVMs: When to use which one? 
( Let's say n and m are the number of features and training samples respectively. If n is large relative to m use log. Reg. or SVM with linear kernel, If n is small and m is intermediate, SVM with Gaussian kernel, If n is small and m is massive, Create or add more fetaures then use log. Reg. or SVM without a kernel)

## Bayesian Machine Learning
1. What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
1. Compare and contrast maximum likelihood and maximum a posteriori estimation.
1. How does Bayesian methods do automatic feature selection?
1. What do you mean by Bayesian regularization?
1. When will you use Bayesian methods instead of Frequentist methods? (Small dataset, large feature set)

## Regularization
1. What is L1 regularization?
1. What is L2 regularization?
1. Compare L1 and L2 regularization.
1. Why does L1 regularization result in sparse models? [here](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)

## Evaluation of Machine Learning systems
1. What are accuracy, sensitivity, specificity, ROC?
1. What are precision and recall?
1. Describe t-test in the context of Machine Learning.

## Clustering
1. Describe the k-means algorithm.
1. What is distortion function? Is it convex or non-convex?
1. Tell me about the convergence of the distortion function.
1. Topic: EM algorithm
1. What is the Gaussian Mixture Model?
1. Describe the EM algorithm intuitively. 
1. What are the two steps of the EM algorithm
1. Compare GMM vs GDA.

## Dimensionality Reduction
1. Why do we need dimensionality reduction techniques? (data compression, speeds up learning algorithm and visualizing data)
1. What do we need PCA and what does it do? (PCA tries to find a lower dimensional surface such the sum of the squared projection error is minimized)
1. What is the difference between logistic regression and PCA?
1. What are the two pre-processing steps that should be applied before doing PCA? (mean normalization and feature scaling)

## Basics of Natural Language Processing
1. What is WORD2VEC?
1. What is t-SNE? Why do we use PCA instead of t-SNE?
1. What is sampled softmax?
1. Why is it difficult to train a RNN with SGD?
1. How do you tackle the problem of exploding gradients? (By gradient clipping)
1. What is the problem of vanishing gradients? (RNN doesn't tend to remember much things from the past)
1. How do you tackle the problem of vanishing gradients? (By using LSTM)
1. Explain the memory cell of a LSTM. (LSTM allows forgetting of data and using long memory when appropriate.)
1. What type of regularization do one use in LSTM?
1. What is Beam Search?
1. How to automatically caption an image? (CNN + LSTM)

## Miscellaneous
1. What is the difference between loss function, cost function and objective function?
